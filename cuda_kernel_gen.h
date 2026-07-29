// CUDA kernel source generation. Pure string building with no CUDA dependency,
// so it compiles anywhere -- only cuda_solver.cpp needs the toolkit.
//
// This is the CUDA counterpart of generateOptimizedShader() in metal_solver.mm.
// The algorithm is identical; see that file's header comment for why it is shaped
// this way and, more importantly, for the list of "optimizations" that measured as
// no-ops and should not be re-attempted.
//
// Two Windows-specific hazards drove type choices here:
//
//   * `long` is 32-bit on Windows (LLP64) while Metal's `long` is 64-bit. Every
//     64-bit value below is `long long` with an `LL` suffix. A stray `long` would
//     silently compute a different hash rather than fail to compile.
//   * Signed 32-bit overflow is UB, and the column hash relies on wraparound. The
//     x multiply and per-block addend are done in `unsigned` and cast back, so the
//     wrap is defined rather than at the optimizer's discretion.
#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "solver_common.h"

// ---------------------------------------------------------------------------
// Reference kernel: a direct transcription of the original algorithm, kept for
// differential testing only. Deliberately unoptimized and structurally identical
// to the Metal reference so the two backends verify against the same semantics.
// MAX_RESULTS is prepended by the caller.
// ---------------------------------------------------------------------------
static const char* kReferenceKernelSrc = R"(
struct RotationInfo { int x, y, z, rotation, is_side; };
struct Uniforms { int x_min, x_max, y_min, y_max, z_min, z_max, num_blocks; };
struct MatchResult { int x, y, z; };

__device__ __forceinline__ int get_texture_fast(long long part_xz, int y, int mod_val) {
    long long l = part_xz ^ (long long)y;
    l = l * l * 42317861LL + l * 11LL;
    long long seed = (l >> 16) ^ 0x5DEECE66DLL;
    seed = (seed * 0x5DEECE66DLL + 11LL) & ((1LL << 48) - 1LL);
    int next = (int)(seed >> 17);
    int res = next >> 29;
    int rem = res % mod_val;
    return rem < 0 ? rem + mod_val : rem;
}

extern "C" __global__ void search_textures(
    const RotationInfo* blocks,
    Uniforms uniforms,
    int* result_count,
    MatchResult* results)
{
    int x = uniforms.x_min + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int z = uniforms.z_min + (int)(blockIdx.y * blockDim.y + threadIdx.y);

    if (x > uniforms.x_max || z > uniforms.z_max) {
        return;
    }

    int num_blocks = uniforms.num_blocks;

    RotationInfo b0 = blocks[0];
    int mod0 = b0.is_side ? 2 : 4;

    for (int y = uniforms.y_min; y <= uniforms.y_max; y++) {
        long long l0 = (long long)(int)((unsigned)(x + b0.x) * 3129871u)
                     ^ ((long long)(z + b0.z) * 116129781LL);
        if (b0.rotation != get_texture_fast(l0, y + b0.y, mod0)) {
            continue;
        }

        bool match = true;
        for (int i = 1; i < num_blocks; i++) {
            RotationInfo b = blocks[i];
            int mod_val = b.is_side ? 2 : 4;
            long long l_b = (long long)(int)((unsigned)(x + b.x) * 3129871u)
                          ^ ((long long)(z + b.z) * 116129781LL);
            if (b.rotation != get_texture_fast(l_b, y + b.y, mod_val)) {
                match = false;
                break;
            }
        }

        if (match) {
            int idx = atomicAdd(result_count, 1);
            if (idx < MAX_RESULTS) {
                results[idx].x = x;
                results[idx].y = y;
                results[idx].z = z;
            }
        }
    }
}
)";

// ---------------------------------------------------------------------------
// Optimized kernel. The formation is baked in as literals so the block chain
// unrolls with no memory loads. Block 0 is forward-differenced across a 64-aligned
// y segment using four strided streams; hits are recorded into a 64-bit mask and
// only survivors descend into the block chain.
// ---------------------------------------------------------------------------
// Design knobs. Defaults are the measured Ada (RTX 4070 SUPER, sm_89) winner.
//
// These deliberately differ from the Metal build, which is the same algorithm:
// Metal ships bitmask=true because deferring the block chain past a per-segment
// survivor mask was its single biggest win (1.25x -> 1.94x on an M1). On Ada the
// identical change is a 3.5x LOSS -- measured against the reference kernel on a
// 6001^2 x 291 box:
//
//     reference                          0.1188 s   1.00x
//     specialized only                   0.0661 s   1.80x
//     + fwd-diff, 1 stream               0.0592 s   2.01x
//     + fwd-diff, 4 streams              0.0569 s   2.09x   <- shipped
//     any of the above + bitmask         ~0.199 s   0.60x
//
// It is not register pressure: at 1 stream both variants use 52 registers. The
// likely cause is `1ULL << (dOff ^ (d + K))` -- a variable 64-bit shift executed
// 64 times per segment, and 64-bit shifts are emulated on NVIDIA. Whatever the
// mechanism, do not "restore" the bitmask here to match Metal without measuring.
struct KernelOptions {
    bool fwdDiff = true;   // forward-difference block 0 across a segment
    int streams = 4;       // ILP width of the forward-difference loop (1, 2 or 4)
    bool bitmask = false;  // defer the block chain via a per-segment survivor mask
    int regBlocks = kRegBlocks;
    int launchBounds = 0;  // emit __launch_bounds__(n) to cap registers; 0 = off
};

inline std::string generateOptimizedKernel(const std::vector<RotationInfo>& blocks,
                                           int maxResults,
                                           const KernelOptions& opt = KernelOptions()) {
    const int n = (int)blocks.size();
    const int regWanted = (opt.regBlocks < 1) ? 1 : opt.regBlocks;
    const int nReg = (n < regWanted) ? n : regWanted;
    const int streams = (opt.streams == 1 || opt.streams == 2) ? opt.streams : 4;

    // Per-block addends. (x + b.x) * kMulX distributes in Z/2^32 and
    // (long long)(z + b.z) * kMulZ distributes in Z/2^64, so these are exact.
    // The 32-bit one is emitted as an unsigned hex literal to sidestep both
    // signed-overflow UB and the INT_MIN-literal problem.
    auto bxc = [&](int i) { return (uint32_t)((int32_t)blocks[i].x * kMulX); };
    auto bzc = [&](int i) { return (int64_t)blocks[i].z * kMulZ; };
    auto mask = [&](int i) { return blocks[i].is_side ? 1 : 3; };
    auto yoff = [&](int i) { return blocks[i].y; };

    auto hex32 = [](uint32_t v) {
        std::ostringstream o;
        o << "0x" << std::hex << v << std::dec << "u";
        return o.str();
    };
    // pxz for block i, computed from the column bases px/pz.
    auto pxzExpr = [&](int i) {
        std::ostringstream o;
        o << "(long long)(int)((unsigned)px + " << hex32(bxc(i)) << ") ^ (pz + "
          << bzc(i) << "LL)";
        return o.str();
    };

    std::ostringstream s;
    s << "#define MAX_RESULTS " << maxResults << "\n\n"
         "struct Uniforms { int x_min, x_max, y_min, y_max, z_min, z_max, num_blocks; };\n"
         "struct MatchResult { int x, y, z; };\n\n"
         "#define A   " << kHashA << "LL\n"
         "#define LCG 0x5DEECE66DLL\n\n"
         "__device__ __forceinline__ int ld_max(int a, int b) { return a > b ? a : b; }\n"
         "__device__ __forceinline__ int ld_min(int a, int b) { return a < b ? a : b; }\n\n"
         "// Bits 46-47 of the stepped LCG state are exactly vanilla nextInt(2|4).\n"
         "// Masking to 48 bits is unnecessary: bits 46-47 sit below bit 48.\n"
         "// The >> 16 assumes an arithmetic shift, which holds on every NVIDIA target.\n"
         "__device__ __forceinline__ int rot_of(long long l, long long m) {\n"
         "    long long s = ((l >> 16) ^ LCG) * LCG + 11LL;\n"
         "    return (int)((s >> 46) & m);\n"
         "}\n\n"
         "__device__ __forceinline__ int rot_at(long long pxz, int y, long long m) {\n"
         "    long long l = pxz ^ (long long)y;\n"
         "    l = l * (l * A + 11LL);\n"
         "    return rot_of(l, m);\n"
         "}\n\n";

    // Blocks 1..n-1, evaluated only for block-0 survivors. Passed as individual
    // scalars, not an array, so they stay in registers.
    std::string params, args;
    for (int i = 1; i < nReg; i++) {
        params += "long long pxz" + std::to_string(i) + ", ";
        args += "pxz" + std::to_string(i) + ", ";
    }
    params += "int px, long long pz, int y";
    args += "px, pz, y";

    const bool needRest = n > 1;
    const std::string restCall = needRest ? ("check_rest(" + args + ")") : "true";

    if (needRest) {
        s << "__device__ __forceinline__ bool check_rest(" << params << ") {\n";
        if (n <= nReg) s << "    (void)px; (void)pz;\n";
        for (int i = 1; i < n; i++) {
            const std::string expr =
                (i < nReg) ? ("pxz" + std::to_string(i)) : ("(" + pxzExpr(i) + ")");
            // The hot loop carries y already offset by block 0's b.y, so the other
            // blocks need their offset expressed relative to block 0.
            const int rel = yoff(i) - yoff(0);
            std::string yexpr = "y";
            if (rel > 0) yexpr = "(y + " + std::to_string(rel) + ")";
            else if (rel < 0) yexpr = "(y - " + std::to_string(-rel) + ")";
            s << "    if (rot_at(" << expr << ", " << yexpr << ", " << mask(i)
              << "LL) != " << blocks[i].rotation << ") return false;\n";
        }
        s << "    return true;\n}\n\n";
    }

    s << "#define EMIT { \\\n"
         "    int idx = atomicAdd(rcount, 1); \\\n"
         "    if (idx < MAX_RESULTS) { results[idx].x = x; results[idx].y = y; "
         "results[idx].z = z; } }\n\n";

    // Two ways to handle a block-0 hit inside the forward-difference loop.
    //
    // FMARK records it into a per-segment mask so the block chain runs afterwards
    // over survivors only. That avoids warp divergence -- a warp is 32 threads and
    // block 0 passes 1 in 4, so nearly every warp has a survivor and would run the
    // chain regardless -- but it keeps the mask and all stream state live at once.
    // FTEST descends immediately, which diverges but frees registers sooner.
    // Which wins is hardware-dependent; measure, do not assume.
    s << "#define FMARK(LV, K) \\\n"
         "    if (rot_of(LV, " << mask(0) << "LL) == " << blocks[0].rotation << ") \\\n"
         "        m |= 1ULL << (dOff ^ (d + (K)));\n\n";
    s << "#define FTEST(LV, K) \\\n"
         "    if (rot_of(LV, " << mask(0) << "LL) == " << blocks[0].rotation << ") { \\\n"
         "        int y = ybase + (dOff ^ (d + (K))); \\\n"
         "        if (" << restCall << ") EMIT \\\n"
         "    }\n\n";

    const int seg = kSegment;
    const int segMask = seg - 1;
    const int y0 = yoff(0);

    s << "extern \"C\" __global__ void ";
    if (opt.launchBounds > 0) s << "__launch_bounds__(" << opt.launchBounds << ") ";
    s << "search_textures(\n"
         "    Uniforms U,\n"
         "    int* rcount,\n"
         "    MatchResult* results)\n"
         "{\n"
         "    int x = U.x_min + (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
         "    int z = U.z_min + (int)(blockIdx.y * blockDim.y + threadIdx.y);\n"
         "    if (x > U.x_max || z > U.z_max) return;\n\n"
         "    int       px = (int)((unsigned)x * " << kMulX << "u);\n"
         "    long long pz = (long long)z * " << kMulZ << "LL;\n\n";
    for (int i = 0; i < nReg; i++) {
        s << "    long long pxz" << i << " = " << pxzExpr(i) << ";\n";
    }

    s << "\n"
         "    int ylo = U.y_min + " << y0 << ";\n"
         "    int yhi = U.y_max + " << y0 << ";\n";

    const std::string directLoopBody =
        "            int y = ybase + t;\n"
        "            if (rot_at(pxz0, y, " + std::to_string(mask(0)) +
        "LL) != " + std::to_string(blocks[0].rotation) + ") continue;\n"
        "            if (" + restCall + ") EMIT\n";

    if (!opt.fwdDiff) {
        // Plain specialized loop: literals baked in and the chain unrolled, but no
        // segment arithmetic. Fewest live registers of any variant.
        s << "    for (int y = ylo; y <= yhi; y++) {\n"
             "        if (rot_at(pxz0, y, " << mask(0) << "LL) != " << blocks[0].rotation
          << ") continue;\n"
             "        if (" << restCall << ") EMIT\n"
             "    }\n"
             "}\n"
             "#undef FMARK\n#undef FTEST\n#undef EMIT\n";
        return s.str();
    }

    s << "\n"
         "    // Segments of the (offset) y axis, " << seg << " wide and " << seg
      << "-aligned;\n"
         "    // & ~" << segMask << " floors toward -inf, so negative y works unchanged.\n"
         "    for (int ybase = (ylo & ~" << segMask << "); ybase <= yhi; ybase += " << seg
      << ") {\n"
         "        int tlo = ld_max(0, ylo - ybase);\n"
         "        int thi = ld_min(" << segMask << ", yhi - ybase);\n\n"
         "        if (tlo == 0 && thi == " << segMask << ") {\n"
         "            // Full segment: forward-difference, no multiplies, no range test.\n"
         "            long long W    = pxz0 ^ (long long)ybase;\n"
         "            long long B    = W & ~" << segMask << "LL;\n"
         "            int       dOff = (int)(W & " << segMask << "LL);\n\n"
         "            long long Q = 2LL * A * B + 11LL;\n"
         "            long long P = B * (B * A + 11LL);\n\n";

    // Stream k of N walks d = k, k+N, k+2N, ...
    //   l_k(0)     = A*k^2 + Q*k + P
    //   delta_k(0) = A*(2kN + N^2) + N*Q
    //   delta increment = 2*A*N^2
    const long long N = streams;
    for (int k = 0; k < streams; k++) {
        s << "            long long l" << k << " = P";
        if (k > 0) s << " + " << k << "LL*Q + " << (long long)k * k << "LL*A";
        s << ";\n";
    }
    for (int k = 0; k < streams; k++) {
        s << "            long long e" << k << " = " << (2LL * k * N + N * N) << "LL*A + "
          << N << "LL*Q;\n";
    }
    s << "            const long long DD = " << (2LL * N * N) << "LL * A;\n\n";

    if (opt.bitmask) s << "            unsigned long long m = 0;\n";
    s << "            for (int d = 0; d < " << seg << "; d += " << streams << ") {\n"
         "               ";
    for (int k = 0; k < streams; k++) {
        s << (opt.bitmask ? " FMARK(l" : " FTEST(l") << k << ", " << k << ")";
    }
    s << "\n";
    for (int k = 0; k < streams; k++) {
        s << "                l" << k << " += e" << k << "; e" << k << " += DD;\n";
    }
    s << "            }\n";

    if (opt.bitmask) {
        s << "\n            // Only block-0 survivors reach the block chain.\n"
             "            while (m) {\n"
             "                int y = ybase + (__ffsll((long long)m) - 1);\n"
             "                m &= m - 1;\n"
             "                if (" << restCall << ") EMIT\n"
             "            }\n";
    }

    s << "        } else {\n"
         "            // Partial segment (at most the first and last): direct path, so a\n"
         "            // short y range never pays for a whole segment of wasted steps.\n"
         "            for (int t = tlo; t <= thi; t++) {\n"
      << directLoopBody
      << "            }\n"
         "        }\n"
         "    }\n"
         "}\n"
         "#undef FMARK\n"
         "#undef FTEST\n"
         "#undef EMIT\n";

    return s.str();
}
