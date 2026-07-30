#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstring>
#include <sstream>

#include "solver_common.h"

// ---------------------------------------------------------------------------
// Kernel generation.
//
// The formation is baked into the source as literals so the block chain unrolls
// into straight-line code with no constant-buffer loads. Two rewrites carry the
// speedup over a direct transcription of the vanilla algorithm:
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
// k=3 1.36x, k=5 slower than the unoptimized original -- each extra masked block
// costs a full 64 multiplies per segment but removes only a handful of survivors).
//
// NOTE: the CUDA backend runs the same algorithm but deliberately ships with the
// bitmask DISABLED -- on Ada it is a 3.5x loss rather than a win. See the
// KernelOptions comment in cuda_kernel_gen.h. Neither default should be copied to
// the other backend without re-measuring there.
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

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> buildPipeline(id<MTLDevice> device,
                                                 const std::string& src) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:src.c_str()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        std::cerr << "Failed to compile shader: "
                  << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@"search_textures"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                            error:&error];
    if (!pso) {
        std::cerr << "Failed to create pipeline: "
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
                      Uniforms bounds,
                      int maxResults,
                      int yOffsetFixup,
                      bool showProgress,
                      std::vector<MatchResult>& outResults,
                      int64_t& outCount,
                      double& outSeconds) {
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

static void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [--rotate <direction>] [x_min x_max y_min y_max z_min z_max]\n\n"
              << "Arguments:\n"
              << "  x_min x_max  Search bounds for X coordinates (default: -6000 6000)\n"
              << "  y_min y_max  Search bounds for Y coordinates (default: 60 256)\n"
              << "  z_min z_max  Search bounds for Z coordinates (default: -6000 6000)\n\n"
              << "Options:\n"
              << "  --rotate <dir>  Cardinal direction for North (north, east, south, west, unknown)\n"
              << "                  If 'unknown', searches all 4 cardinal rotations sequentially.\n"
              << "                  (default: north)\n"
              << "  -h, --help      Show this help\n\n"
              << "Environment:\n"
              << "  LODESTONE_TG_H   Threadgroup height for tuning (default 8)\n\n"
              << "Example:\n"
              << "  " << prog << " --rotate east -200 200 60 128 -200 200\n"
              << "  " << prog << " --rotate unknown -6000 6000 60 256 -6000 6000\n";
}

struct CmdArgs {
    RotateDir rotate = RotateDir::NORTH;
    int x_min = -6000;
    int x_max = 6000;
    int y_min = 60;
    int y_max = 256;
    int z_min = -6000;
    int z_max = 6000;
    bool showHelp = false;
};

static CmdArgs parseArgs(int argc, char* argv[]) {
    CmdArgs args;
    std::vector<std::string> positionals;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            args.showHelp = true;
        } else if (arg == "--rotate" || arg == "-r") {
            if (i + 1 < argc) {
                args.rotate = parseRotateDir(argv[++i]);
            }
        } else if (arg.rfind("--rotate=", 0) == 0) {
            args.rotate = parseRotateDir(arg.substr(9));
        } else {
            positionals.push_back(arg);
        }
    }

    if (positionals.size() >= 1) args.x_min = atoi(positionals[0].c_str());
    if (positionals.size() >= 2) args.x_max = atoi(positionals[1].c_str());
    if (positionals.size() >= 3) args.y_min = atoi(positionals[2].c_str());
    if (positionals.size() >= 4) args.y_max = atoi(positionals[3].c_str());
    if (positionals.size() >= 5) args.z_min = atoi(positionals[4].c_str());
    if (positionals.size() >= 6) args.z_max = atoi(positionals[5].c_str());

    return args;
}

int main(int argc, char* argv[]) {
    CmdArgs args = parseArgs(argc, argv);
    if (args.showHelp) {
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

        std::vector<RotationInfo> baseFormation = loadFormation();

        std::vector<RotateDir> dirsToSearch;
        if (args.rotate == RotateDir::UNKNOWN) {
            dirsToSearch = {RotateDir::NORTH, RotateDir::EAST, RotateDir::SOUTH, RotateDir::WEST};
        } else {
            dirsToSearch = {args.rotate};
        }

        int totalMatches = 0;
        double totalSeconds = 0;

        for (RotateDir dir : dirsToSearch) {
            if (args.rotate == RotateDir::UNKNOWN) {
                std::cout << "\n=== Searching Rotation: " << rotateDirName(dir) << " ===" << std::endl;
            }

            std::vector<RotationInfo> formation = rotateFormation(baseFormation, dir);
            orderByRejectionPower(formation);

            Uniforms uniforms = {
                .x_min = args.x_min,
                .x_max = args.x_max,
                .y_min = args.y_min,
                .y_max = args.y_max,
                .z_min = args.z_min,
                .z_max = args.z_max,
                .num_blocks = (int)formation.size()
            };

            if (!validateBounds(uniforms, formation)) continue;

            id<MTLComputePipelineState> pso =
                buildPipeline(device, generateOptimizedShader(formation, kDefaultMaxResults));
            if (!pso) return 1;

            std::vector<MatchResult> results;
            int64_t count = 0;
            double seconds = 0;
            if (!runSearch(device, commandQueue, pso, uniforms, kDefaultMaxResults,
                           blockZeroYOffset(formation), true, results, count, seconds)) {
                return 1;
            }

            totalSeconds += seconds;
            totalMatches += (int)count;

            if (args.rotate == RotateDir::UNKNOWN) {
                std::cout << "Found " << count << " match(es) for rotation [" << rotateDirName(dir) << "]:" << std::endl;
            } else {
                std::cout << "\nFound " << count << " match(es):" << std::endl;
            }

            for (const auto& r : results) {
                std::cout << "X: " << r.x << " Y: " << r.y << " Z: " << r.z << std::endl;
            }
            if (count > kDefaultMaxResults) {
                std::cout << "(only the first " << kDefaultMaxResults
                          << " matches were recorded)" << std::endl;
            }

            if (args.rotate != RotateDir::UNKNOWN) {
                std::cout << "\nMetal GPU execution time: " << seconds << " seconds" << std::endl;
            }
        }

        if (args.rotate == RotateDir::UNKNOWN) {
            std::cout << "\nTotal execution time across all 4 rotations: " << totalSeconds << " seconds" << std::endl;
        }
    }
    return 0;
}

