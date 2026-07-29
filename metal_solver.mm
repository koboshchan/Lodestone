#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

struct RotationInfo {
    int x;
    int y;
    int z;
    int rotation;
    int is_side;
};

struct Uniforms {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    int z_min;
    int z_max;
    int num_blocks;
};

struct MatchResult {
    int x;
    int y;
    int z;
};

static bool operator<(const MatchResult& a, const MatchResult& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}
static bool operator==(const MatchResult& a, const MatchResult& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Vanilla coordinate hash constants.
static const int32_t kMulX = 3129871;
static const int64_t kMulZ = 116129781LL;
static const int64_t kHashA = 42317861LL;

// Number of per-block column hashes kept in registers. Blocks past this index are
// reached with probability <= 4^-kRegBlocks, so recomputing them inline (2 adds +
// an xor) is cheaper than the occupancy cost of holding them.
static const int kRegBlocks = 4;

// Width of a forward-difference y segment. Fixed at 64 because each segment's
// block-0 results are accumulated into a single 64-bit mask. Only full segments
// take the fast path; 64 also keeps waste low on a typical 200-300 tall search,
// where a wider segment would push most of the range onto the direct path.
static const int kSegment = 64;

static const int kDefaultMaxResults = 4096;

// ---------------------------------------------------------------------------
// Reference kernel: the original implementation, preserved for differential
// testing. The only change from the original is that the results-buffer capacity
// is a #define instead of the literal 100, so --verify can capture large match
// sets. The arithmetic is untouched.
// ---------------------------------------------------------------------------
static const char* kReferenceShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

struct RotationInfo {
    int x;
    int y;
    int z;
    int rotation;
    int is_side;
};

struct Uniforms {
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    int z_min;
    int z_max;
    int num_blocks;
};

struct MatchResult {
    int x;
    int y;
    int z;
};

inline int get_texture_fast(long part_xz, int x, int y, int z, int mod_val) {
    long l = part_xz ^ (long)y;
    l = l * l * 42317861L + l * 11L;
    long seed = (l >> 16) ^ 0x5DEECE66DL;
    seed = (seed * 0x5DEECE66DL + 11L) & ((1L << 48) - 1L);
    int next = (int)(seed >> 17);
    int res = next >> 29;
    int rem = res % mod_val;
    return rem < 0 ? rem + mod_val : rem;
}

kernel void search_textures(
    constant RotationInfo* blocks [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    device atomic_int* result_count [[buffer(2)]],
    device MatchResult* results [[buffer(3)]],
    uint2 thread_position_in_grid [[thread_position_in_grid]]
) {
    int x = uniforms.x_min + (int)thread_position_in_grid.x;
    int z = uniforms.z_min + (int)thread_position_in_grid.y;

    if (x > uniforms.x_max || z > uniforms.z_max) {
        return;
    }

    int num_blocks = uniforms.num_blocks;

    RotationInfo b0 = blocks[0];
    int mod0 = b0.is_side ? 2 : 4;

    for (int y = uniforms.y_min; y <= uniforms.y_max; y++) {
        long l0 = (long)((x + b0.x) * 3129871) ^ ((long)(z + b0.z) * 116129781L);
        if (b0.rotation != get_texture_fast(l0, x + b0.x, y + b0.y, z + b0.z, mod0)) {
            continue;
        }

        bool match = true;
        for (int i = 1; i < num_blocks; i++) {
            RotationInfo b = blocks[i];
            int mod_val = b.is_side ? 2 : 4;
            long l_b = (long)((x + b.x) * 3129871) ^ ((long)(z + b.z) * 116129781L);
            if (b.rotation != get_texture_fast(l_b, x + b.x, y + b.y, z + b.z, mod_val)) {
                match = false;
                break;
            }
        }

        if (match) {
            int idx = atomic_fetch_add_explicit(result_count, 1, memory_order_relaxed);
            if (idx < MAX_RESULTS) {
                results[idx] = MatchResult{x, y, z};
            }
        }
    }
}
)";

// ---------------------------------------------------------------------------
// Optimized kernel generation.
//
// The formation is baked into the source as literals so the block chain unrolls
// into straight-line code with no constant-buffer loads. Two rewrites carry the
// speedup, both bit-exact against the reference kernel (see --verify):
//
//   1. Forward differencing. Over a 64-aligned y segment, l = pxz ^ y decomposes
//      to B + d with d a permutation of 0..63, so l(d) = A*d^2 + Q*d + P. A
//      quadratic has a constant second difference, so l advances with two adds
//      and no multiply, leaving only the LCG step. Four streams strided by 4 keep
//      independent LCG multiplies in flight, since the chain is latency-bound.
//
//   2. Deferred block chain. Descending into blocks 1..n-1 inline wastes most of
//      its work: a SIMD group is 32 threads and block 0 passes 1 in 4, so nearly
//      every group has a survivor and executes the chain regardless. Instead,
//      block 0's results for all 64 y values go into a bitmask and only survivors
//      are walked. This was worth more than everything else combined (measured
//      1.25x -> 1.94x on an M1).
//
// Two rewrites that look like wins are deliberately NOT relied on: replacing the
// `% mod_val` chain with a bit extract, and factoring l*l*A + l*11 into
// l*(l*A + 11). Both are written that way below for clarity, but measurement
// showed the Metal compiler already performs them plus the loop-invariant hoist
// of the column hash -- those three variants time identically. Do not expect
// gains from re-doing that kind of local rewrite here.
//
// Also measured and rejected: masking on 2+ blocks instead of 1 (k=2 is 1.74x,
// k=3 1.36x, k=5 slower than the original -- each extra masked block costs a full
// 64 multiplies per segment but removes only a handful of survivors).
// ---------------------------------------------------------------------------
static std::string generateOptimizedShader(const std::vector<RotationInfo>& blocks,
                                           int maxResults) {
    const int n = (int)blocks.size();
    const int nReg = std::min(n, kRegBlocks);

    // Per-block addends. (x + b.x) * kMulX distributes in Z/2^32, and
    // (long)(z + b.z) * kMulZ distributes in Z/2^64, so these are exact.
    auto bxc = [&](int i) { return (int32_t)((int32_t)blocks[i].x * kMulX); };
    auto bzc = [&](int i) { return (int64_t)blocks[i].z * kMulZ; };
    auto mask = [&](int i) { return blocks[i].is_side ? 1 : 3; };

    // A block at b.y != 0 is evaluated at y + b.y; fold that into the segment
    // arithmetic by offsetting the y passed to rot_at.
    auto yoff = [&](int i) { return blocks[i].y; };

    std::ostringstream s;
    s << "#include <metal_stdlib>\n"
         "using namespace metal;\n\n"
         "#define MAX_RESULTS " << maxResults << "\n\n"
         "struct Uniforms { int x_min, x_max, y_min, y_max, z_min, z_max, num_blocks; };\n"
         "struct MatchResult { int x, y, z; };\n\n"
         "constant long A   = " << kHashA << "L;\n"
         "constant long LCG = 0x5DEECE66DL;\n\n"
         "// Bits 46-47 of the stepped LCG state are exactly vanilla nextInt(2|4).\n"
         "// Masking to 48 bits is unnecessary: bits 46-47 sit below bit 48.\n"
         "inline int rot_of(long l, long m) {\n"
         "    long s = ((l >> 16) ^ LCG) * LCG + 11L;\n"
         "    return (int)((s >> 46) & m);\n"
         "}\n\n"
         "inline int rot_at(long pxz, int y, long m) {\n"
         "    long l = pxz ^ (long)y;\n"
         "    l = l * (l * A + 11L);\n"
         "    return rot_of(l, m);\n"
         "}\n\n";

    // Blocks 1..n-1, evaluated only after block 0 passes (~25% of candidates).
    // Passed as individual scalars, not an array, so they stay in registers.
    std::string params, args;
    for (int i = 1; i < nReg; i++) {
        params += "long pxz" + std::to_string(i) + ", ";
        args += "pxz" + std::to_string(i) + ", ";
    }
    params += "int px, long pz, int y";
    args += "px, pz, y";

    const bool needRest = n > 1;
    const std::string restCall = needRest ? ("check_rest(" + args + ")") : "true";

    if (needRest) {
    s << "inline bool check_rest(" << params << ") {\n";
    if (n <= nReg) {
        s << "    (void)px; (void)pz;\n";
    }
    for (int i = 1; i < n; i++) {
        std::string expr;
        if (i < nReg) {
            expr = "pxz" + std::to_string(i);
        } else {
            std::ostringstream e;
            e << "((long)(int)(px + " << bxc(i) << ") ^ (pz + " << bzc(i) << "L))";
            expr = e.str();
        }
        // The hot loop carries y already offset by block 0's b.y, so the other
        // blocks need their offset expressed relative to block 0.
        const int rel = yoff(i) - yoff(0);
        std::string yexpr = "y";
        if (rel > 0) yexpr = "(y + " + std::to_string(rel) + ")";
        else if (rel < 0) yexpr = "(y - " + std::to_string(-rel) + ")";
        s << "    if (rot_at(" << expr << ", " << yexpr << ", " << mask(i) << "L) != "
          << blocks[i].rotation << ") return false;\n";
    }
    s << "    return true;\n"
         "}\n\n";
    }

    s << "#define EMIT { \\\n"
         "    int idx = atomic_fetch_add_explicit(rcount, 1, memory_order_relaxed); \\\n"
         "    if (idx < MAX_RESULTS) results[idx] = MatchResult{x, y, z}; }\n\n";

    // Block 0 drives the hot loop; its y offset is folded into the segment base.
    // Rather than descending into the block chain inline -- which diverges, since
    // with 32 threads per SIMD group and a 1-in-4 pass rate essentially every group
    // has a survivor -- record the hit in a per-segment bitmask and walk survivors
    // afterwards. The masked phase then runs branch-free.
    s << "#define FMARK(LV, K) \\\n"
         "    if (rot_of(LV, " << mask(0) << "L) == " << blocks[0].rotation << ") \\\n"
         "        m |= 1UL << (dOff ^ (d + (K)));\n\n";

    s << "kernel void search_textures(\n"
         "    constant Uniforms& U [[buffer(1)]],\n"
         "    device atomic_int* rcount [[buffer(2)]],\n"
         "    device MatchResult* results [[buffer(3)]],\n"
         "    uint2 tid [[thread_position_in_grid]]\n"
         ") {\n"
         "    int x = U.x_min + (int)tid.x;\n"
         "    int z = U.z_min + (int)tid.y;\n"
         "    if (x > U.x_max || z > U.z_max) return;\n\n"
         "    int  px = x * " << kMulX << ";\n"
         "    long pz = (long)z * " << kMulZ << "L;\n\n";
    for (int i = 0; i < nReg; i++) {
        s << "    long pxz" << i << " = (long)(int)(px + " << bxc(i) << ") ^ (pz + "
          << bzc(i) << "L);\n";
    }

    // Block 0's y offset shifts which y each segment covers.
    const int y0 = yoff(0);
    const int seg = kSegment;
    const int segMask = seg - 1;
    s << "\n"
         "    // Segments of the (offset) y axis, " << seg << " wide and " << seg
      << "-aligned;\n"
         "    // & ~" << segMask << " floors toward -inf, so negative y works unchanged.\n"
         "    int ylo = U.y_min + " << y0 << ";\n"
         "    int yhi = U.y_max + " << y0 << ";\n"
         "    for (int ybase = (ylo & ~" << segMask << "); ybase <= yhi; ybase += " << seg
      << ") {\n"
         "        int tlo = max(0, ylo - ybase);\n"
         "        int thi = min(" << segMask << ", yhi - ybase);\n\n"
         "        if (tlo == 0 && thi == " << segMask << ") {\n"
         "            // Full segment: forward-difference, no multiplies, no range test.\n"
         "            long W    = pxz0 ^ (long)ybase;\n"
         "            long B    = W & ~" << segMask << "L;\n"
         "            int  dOff = (int)(W & " << segMask << "L);\n\n"
         "            long Q = 2L * A * B + 11L;\n"
         "            long P = B * (B * A + 11L);\n\n"
         "            long l0 = P,               l1 = P + Q + A;\n"
         "            long l2 = P + 2L*Q + 4L*A, l3 = P + 3L*Q + 9L*A;\n"
         "            long e0 = 16L*A + 4L*Q,    e1 = 24L*A + 4L*Q;\n"
         "            long e2 = 32L*A + 4L*Q,    e3 = 40L*A + 4L*Q;\n"
         "            const long DD = 32L * A;\n\n"
         "            ulong m = 0;\n"
         "            for (int d = 0; d < " << seg << "; d += 4) {\n"
         "                FMARK(l0, 0) FMARK(l1, 1) FMARK(l2, 2) FMARK(l3, 3)\n"
         "                l0 += e0; e0 += DD;\n"
         "                l1 += e1; e1 += DD;\n"
         "                l2 += e2; e2 += DD;\n"
         "                l3 += e3; e3 += DD;\n"
         "            }\n\n"
         "            // Only block-0 survivors reach the block chain.\n"
         "            while (m) {\n"
         "                int y = ybase + (int)ctz(m);\n"
         "                m &= m - 1;\n"
         "                if (" << restCall << ") EMIT\n"
         "            }\n"
         "        } else {\n"
         "            // Partial segment (at most the first and last): direct path, so a\n"
         "            // short y range never pays for a whole segment of wasted steps.\n"
         "            for (int t = tlo; t <= thi; t++) {\n"
         "                int y = ybase + t;\n"
         "                if (rot_at(pxz0, y, " << mask(0) << "L) != " << blocks[0].rotation
      << ") continue;\n"
         "                if (" << restCall << ") EMIT\n"
         "            }\n"
         "        }\n"
         "    }\n"
         "}\n"
         "#undef FMARK\n"
         "#undef EMIT\n";

    return s.str();
}

// The y recovered inside the kernel is offset by block 0's b.y; undo it on readback.
static int blockZeroYOffset(const std::vector<RotationInfo>& f) { return f[0].y; }

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> buildPipeline(id<MTLDevice> device,
                                                 const std::string& src,
                                                 const char* label) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:src.c_str()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "Failed to compile " << label << " shader: "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@"search_textures"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (!pso) {
        std::cerr << "Failed to create " << label << " pipeline: "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    return pso;
}

// Runs a search, slicing the z axis so each command buffer is short enough to
// report progress. Returns false if the GPU reported an error.
static bool runSearch(id<MTLDevice> device,
                      id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pso,
                      const std::vector<RotationInfo>& formation,
                      Uniforms bounds,
                      int maxResults,
                      int yOffsetFixup,
                      bool showProgress,
                      std::vector<MatchResult>& outResults,
                      int64_t& outCount,
                      double& outSeconds) {
    id<MTLBuffer> blocksBuffer =
        [device newBufferWithBytes:formation.data()
                            length:formation.size() * sizeof(RotationInfo)
                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> uniformsBuffer = [device newBufferWithLength:sizeof(Uniforms)
                                                       options:MTLResourceStorageModeShared];

    int zero = 0;
    id<MTLBuffer> countBuffer = [device newBufferWithBytes:&zero
                                                    length:sizeof(int)
                                                   options:MTLResourceStorageModeShared];

    id<MTLBuffer> resultsBuffer =
        [device newBufferWithLength:(NSUInteger)maxResults * sizeof(MatchResult)
                            options:MTLResourceStorageModeShared];

    const NSUInteger w = pso.threadExecutionWidth;
    NSUInteger h = pso.maxTotalThreadsPerThreadgroup / w;
    if (const char* envH = getenv("LODESTONE_TG_H")) {
        NSUInteger want = (NSUInteger)atoi(envH);
        if (want >= 1 && want * w <= pso.maxTotalThreadsPerThreadgroup) h = want;
    } else {
        h = std::min<NSUInteger>(h, 8);  // lower occupancy pressure than 32x32
    }
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

    const int64_t xSpan = (int64_t)bounds.x_max - bounds.x_min + 1;
    const int64_t ySpan = (int64_t)bounds.y_max - bounds.y_min + 1;
    const int64_t zSpan = (int64_t)bounds.z_max - bounds.z_min + 1;

    // Aim for a few hundred ms of work per command buffer.
    const int64_t kIterPerChunk = 4000000000LL;
    int64_t perRow = std::max<int64_t>(1, xSpan * ySpan);
    int64_t rowsPerChunk = std::max<int64_t>(1, kIterPerChunk / perRow);
    rowsPerChunk = std::min(rowsPerChunk, zSpan);

    auto start = std::chrono::high_resolution_clock::now();

    for (int64_t zStart = bounds.z_min; zStart <= bounds.z_max; zStart += rowsPerChunk) {
        int64_t zEnd = std::min<int64_t>(zStart + rowsPerChunk - 1, bounds.z_max);

        Uniforms chunk = bounds;
        chunk.z_min = (int)zStart;
        chunk.z_max = (int)zEnd;
        memcpy([uniformsBuffer contents], &chunk, sizeof(Uniforms));

        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:blocksBuffer offset:0 atIndex:0];
            [encoder setBuffer:uniformsBuffer offset:0 atIndex:1];
            [encoder setBuffer:countBuffer offset:0 atIndex:2];
            [encoder setBuffer:resultsBuffer offset:0 atIndex:3];

            MTLSize gridSize = MTLSizeMake((NSUInteger)xSpan,
                                           (NSUInteger)(zEnd - zStart + 1), 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            if (commandBuffer.error) {
                std::cerr << "\nGPU command buffer failed: "
                          << [[commandBuffer.error localizedDescription] UTF8String]
                          << std::endl;
                return false;
            }
        }

        if (showProgress) {
            double done = (double)(zEnd - bounds.z_min + 1) / (double)zSpan;
            std::cerr << "\rSearching... " << (int)(done * 100.0) << "%   " << std::flush;
        }
    }

    if (showProgress) std::cerr << "\r                      \r" << std::flush;

    auto end = std::chrono::high_resolution_clock::now();
    outSeconds = std::chrono::duration<double>(end - start).count();

    outCount = *(int*)[countBuffer contents];
    MatchResult* ptr = (MatchResult*)[resultsBuffer contents];
    int64_t stored = std::min<int64_t>(outCount, maxResults);
    outResults.assign(ptr, ptr + stored);
    for (auto& r : outResults) r.y -= yOffsetFixup;
    return true;
}

static std::vector<RotationInfo> loadFormation() {
    std::ifstream file("formation.txt");
    if (!file.is_open()) {
        std::cout << "formation.txt not found, using default Unnamed5 formation." << std::endl;
        return {
            {0, 0, 0, 3, 0}, {1, 0, 0, 1, 0}, {2, 0, 0, 1, 0}, {3, 0, 0, 0, 0}, {4, 0, 0, 3, 0},
            {0, 0, 1, 1, 0}, {1, 0, 1, 1, 0}, {2, 0, 1, 2, 0}, {3, 0, 1, 3, 0}, {4, 0, 1, 2, 0},
            {0, 0, 2, 2, 0}, {1, 0, 2, 2, 0}, {2, 0, 2, 2, 0}, {3, 0, 2, 3, 0}, {4, 0, 2, 3, 0},
            {0, 0, 3, 0, 0}, {1, 0, 3, 1, 0}, {2, 0, 3, 1, 0}, {3, 0, 3, 3, 0}, {4, 0, 3, 3, 0}
        };
    }

    int count = 0;
    file >> count;
    if (count <= 0) {
        std::cerr << "formation.txt declares " << count << " blocks; nothing to search for."
                  << std::endl;
        exit(1);
    }
    std::vector<RotationInfo> formation(count);
    for (int i = 0; i < count; i++) {
        file >> formation[i].x >> formation[i].y >> formation[i].z
             >> formation[i].rotation >> formation[i].is_side;
    }
    if (!file) {
        std::cerr << "formation.txt is truncated: expected " << count << " block rows."
                  << std::endl;
        exit(1);
    }
    std::cout << "Loaded " << count << " block observations from formation.txt" << std::endl;
    return formation;
}

// A conjunction is order-independent, so putting the mod-4 blocks (which reject
// 3 of 4 candidates) ahead of the mod-2 blocks cannot change the result set.
static void orderByRejectionPower(std::vector<RotationInfo>& f) {
    std::stable_partition(f.begin(), f.end(),
                          [](const RotationInfo& b) { return b.is_side == 0; });
}

// The optimized kernel folds (z + b.z) into pz + b.z*kMulZ, which differs from
// the reference only if (z + b.z) overflows int32. Reject those bounds.
static bool validateBounds(const Uniforms& u, const std::vector<RotationInfo>& f) {
    if (u.x_min > u.x_max || u.y_min > u.y_max || u.z_min > u.z_max) {
        std::cerr << "Empty search range." << std::endl;
        return false;
    }
    int64_t zLo = u.z_min, zHi = u.z_max, yLo = u.y_min, yHi = u.y_max;
    for (const auto& b : f) {
        zLo = std::min<int64_t>(zLo, (int64_t)u.z_min + b.z);
        zHi = std::max<int64_t>(zHi, (int64_t)u.z_max + b.z);
        yLo = std::min<int64_t>(yLo, (int64_t)u.y_min + b.y);
        yHi = std::max<int64_t>(yHi, (int64_t)u.y_max + b.y);
    }
    const int64_t kLimit = 2000000000LL;
    if (zLo < -kLimit || zHi > kLimit || yLo < -kLimit || yHi > kLimit) {
        std::cerr << "Search bounds too large; y and z (plus formation offsets) must "
                     "stay within +/-2e9." << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Differential verification against the reference kernel
// ---------------------------------------------------------------------------
struct VerifyCase {
    const char* name;
    std::vector<RotationInfo> formation;
    Uniforms bounds;
};

static int runVerify(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    const int maxResults = 1 << 22;  // 4.2M entries, 50 MB

    // 1- and 2-block formations match ~25% and ~6% of all coordinates, so these
    // compare the hash at effectively every point in the box. A rare 20-block
    // formation would only ever prove empty == empty.
    std::vector<RotationInfo> one   = {{0, 0, 0, 0, 0}};
    std::vector<RotationInfo> two   = {{0, 0, 0, 2, 0}, {1, 0, 0, 3, 0}};
    std::vector<RotationInfo> side  = {{0, 0, 0, 1, 1}};                  // mod 2 path
    std::vector<RotationInfo> vert  = {{0, 3, 0, 1, 0}, {0, -2, 1, 2, 0}}; // b.y != 0

    // Eight copies of one block all test the identical condition, so the match
    // rate stays at 25% while the full 8-deep unrolled chain runs -- including
    // blocks 4..7, which take the inline-recompute path rather than registers.
    std::vector<RotationInfo> deep(8, RotationInfo{0, 0, 0, 0, 0});

    // Five distinct blocks: exercises the inline path with a real nonzero offset.
    std::vector<RotationInfo> five = {{0, 0, 0, 1, 0}, {1, 0, 0, 2, 0}, {0, 0, 1, 3, 0},
                                      {2, 1, 0, 0, 0}, {-1, 0, -2, 1, 0}};

    std::vector<RotationInfo> real  = loadFormation();

    std::vector<VerifyCase> cases = {
        {"1 block, aligned y",        one,  {-64,  63,   0, 255, -64,  63, 0}},
        {"1 block, straddles origin", one,  {-96,  31, -20,  90, -37,  58, 0}},
        {"1 block, negative y",       one,  {-64,  63, -300, -1, -64,  63, 0}},
        {"1 block, spans segment",    one,  {-64,  63,   -3, 300, -64, 63, 0}},
        {"1 block, sub-segment y",    one,  {-128, 127,  70,  72, -128, 127, 0}},
        {"1 block, single y",         one,  {-128, 127,  64,  64, -128, 127, 0}},
        {"2 blocks",                  two,  {-256, 255,  -5, 260, -128, 127, 0}},
        {"is_side (mod 2)",           side, {-64,   63,   0, 255,  -64,  63, 0}},
        {"vertical offsets",          vert, {-128, 127,  -9, 300, -128, 127, 0}},
        {"8 blocks (deep chain)",     deep, {-64,   63,   0, 255,  -64,  63, 0}},
        {"5 blocks (inline path)",    five, {-512, 511,  -7, 260, -512, 511, 0}},
        {"formation.txt",             real, {-4200, -3900, 0, 290, -1700, -1300, 0}},
    };

    id<MTLComputePipelineState> refPso = buildPipeline(
        device,
        "#define MAX_RESULTS " + std::to_string(maxResults) + "\n" + kReferenceShaderSrc,
        "reference");
    if (!refPso) return 1;

    int failures = 0;
    int vacuous = 0;
    for (auto& c : cases) {
        c.bounds.num_blocks = (int)c.formation.size();

        std::vector<RotationInfo> optFormation = c.formation;
        orderByRejectionPower(optFormation);

        id<MTLComputePipelineState> optPso = buildPipeline(
            device, generateOptimizedShader(optFormation, maxResults), "optimized");
        if (!optPso) return 1;

        std::vector<MatchResult> refOut, optOut;
        int64_t refCount = 0, optCount = 0;
        double refSecs = 0, optSecs = 0;

        if (!runSearch(device, queue, refPso, c.formation, c.bounds, maxResults, 0,
                       false, refOut, refCount, refSecs)) return 1;
        if (!runSearch(device, queue, optPso, optFormation, c.bounds, maxResults,
                       blockZeroYOffset(optFormation), false, optOut, optCount, optSecs))
            return 1;

        std::sort(refOut.begin(), refOut.end());
        std::sort(optOut.begin(), optOut.end());

        bool overflow = refCount > maxResults || optCount > maxResults;
        bool ok = !overflow && refCount == optCount && refOut == optOut;

        // Zero matches on both kernels agrees trivially and proves nothing; say so
        // rather than reporting a pass that carries no information.
        const char* verdict = !ok ? "  FAIL  " : (refCount == 0 ? "  VOID  " : "  PASS  ");
        if (ok && refCount == 0) vacuous++;

        std::cout << verdict << c.name << "  (" << refCount << " matches";
        if (refSecs > 0 && optSecs > 0) {
            std::cout << ", " << (refSecs / optSecs) << "x";
        }
        std::cout << ")" << std::endl;

        if (!ok) {
            failures++;
            if (overflow) {
                std::cout << "        result buffer overflowed (" << refCount << " vs cap "
                          << maxResults << "); shrink the case box" << std::endl;
            } else if (refCount != optCount) {
                std::cout << "        count mismatch: reference " << refCount
                          << ", optimized " << optCount << std::endl;
            }
            size_t shown = 0;
            for (size_t i = 0; i < std::max(refOut.size(), optOut.size()) && shown < 5; i++) {
                bool differ = i >= refOut.size() || i >= optOut.size() ||
                              !(refOut[i] == optOut[i]);
                if (!differ) continue;
                std::cout << "        [" << i << "] ref=";
                if (i < refOut.size())
                    std::cout << refOut[i].x << "," << refOut[i].y << "," << refOut[i].z;
                else std::cout << "-";
                std::cout << "  opt=";
                if (i < optOut.size())
                    std::cout << optOut[i].x << "," << optOut[i].y << "," << optOut[i].z;
                else std::cout << "-";
                std::cout << std::endl;
                shown++;
            }
        }
    }

    std::cout << std::endl;
    if (failures == 0) {
        std::cout << "All verification cases passed." << std::endl;
    } else {
        std::cout << failures << " case(s) FAILED." << std::endl;
    }
    if (vacuous > 0) {
        std::cout << vacuous << " case(s) marked VOID: both kernels found nothing, so "
                     "they agree only trivially." << std::endl;
    }
    return failures == 0 ? 0 : 1;
}

static void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [x_min x_max y_min y_max z_min z_max]\n"
              << "       " << prog << " --verify\n\n"
              << "Arguments:\n"
              << "  x_min x_max  Search bounds for X coordinates (default: -6000 6000)\n"
              << "  y_min y_max  Search bounds for Y coordinates (default: 60 256)\n"
              << "  z_min z_max  Search bounds for Z coordinates (default: -6000 6000)\n\n"
              << "Options:\n"
              << "  --verify     Diff the optimized kernel against the reference kernel\n"
              << "  -h, --help   Show this help\n\n"
              << "Environment:\n"
              << "  LODESTONE_TG_H   Threadgroup height for tuning (default 8)\n\n"
              << "Example:\n"
              << "  " << prog << " -200 200 60 128 -200 200\n"
              << "  " << prog << " -6000 6000 60 256 -6000 6000\n";
}

int main(int argc, char* argv[]) {
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printHelp(argv[0]);
        return 0;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device." << std::endl;
            return 1;
        }
        std::cout << "Using Metal GPU Device: " << [[device name] UTF8String] << std::endl;
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];


        if (argc > 1 && strcmp(argv[1], "--verify") == 0) {
            return runVerify(device, commandQueue);
        }

        std::vector<RotationInfo> formation = loadFormation();
        orderByRejectionPower(formation);

        Uniforms uniforms = {
            .x_min = (argc > 1) ? atoi(argv[1]) : -6000,
            .x_max = (argc > 2) ? atoi(argv[2]) : 6000,
            .y_min = (argc > 3) ? atoi(argv[3]) : 60,
            .y_max = (argc > 4) ? atoi(argv[4]) : 256,
            .z_min = (argc > 5) ? atoi(argv[5]) : -6000,
            .z_max = (argc > 6) ? atoi(argv[6]) : 6000,
            .num_blocks = (int)formation.size()
        };

        if (!validateBounds(uniforms, formation)) return 1;

        id<MTLComputePipelineState> pso = buildPipeline(
            device, generateOptimizedShader(formation, kDefaultMaxResults), "optimized");
        if (!pso) return 1;

        std::vector<MatchResult> results;
        int64_t count = 0;
        double seconds = 0;
        if (!runSearch(device, commandQueue, pso, formation, uniforms, kDefaultMaxResults,
                       blockZeroYOffset(formation), true, results, count, seconds)) {
            return 1;
        }

        std::cout << "\nFound " << count << " match(es):" << std::endl;
        for (const auto& r : results) {
            std::cout << "X: " << r.x << " Y: " << r.y << " Z: " << r.z << std::endl;
        }
        if (count > kDefaultMaxResults) {
            std::cout << "(only the first " << kDefaultMaxResults
                      << " matches were recorded)" << std::endl;
        }

        std::cout << "\nMetal GPU execution time: " << seconds << " seconds" << std::endl;
    }
    return 0;
}
