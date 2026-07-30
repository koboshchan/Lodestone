// CUDA texture-rotation solver -- native Windows counterpart of metal_solver.mm.
//
// The kernel is generated as source at runtime and compiled with NVRTC, exactly
// as the Metal build compiles its shader from a string. A consequence worth
// knowing: nothing here is device code, so this file is built by cl.exe alone --
// nvcc is not required, and the target architecture is discovered from the device
// rather than baked in at build time.
//
// Uses the CUDA driver API exclusively. Mixing in the runtime API (cudaMalloc and
// friends) would put two context-management schemes in the same process, which is
// a well-known source of subtle breakage.
#include <cuda.h>
#include <nvrtc.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_kernel_gen.h"
#include "solver_common.h"

// ---------------------------------------------------------------------------
// Error plumbing
// ---------------------------------------------------------------------------
static bool cuOk(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    std::cerr << "CUDA error in " << what << ": " << (name ? name : "?") << " -- "
              << (desc ? desc : "no description") << std::endl;
    return false;
}

#define CU_TRY(call)                                   \
    do {                                               \
        if (!cuOk((call), #call)) return false;        \
    } while (0)

// ---------------------------------------------------------------------------
// NVRTC compilation
// ---------------------------------------------------------------------------
struct Gpu {
    CUdevice device = 0;
    CUcontext context = nullptr;
    int major = 0;
    int minor = 0;
    std::string name;
};

// Compiles to a CUBIN for the exact architecture when possible, falling back to
// PTX (JIT-compiled at module load) on older NVRTC or unsupported targets.
static bool compileKernel(const std::string& src, const Gpu& gpu,
                          std::vector<char>& image) {
    const std::string sm = std::to_string(gpu.major) + std::to_string(gpu.minor);

    for (int attempt = 0; attempt < 2; attempt++) {
        const bool cubin = (attempt == 0);
        const std::string arch =
            "--gpu-architecture=" + std::string(cubin ? "sm_" : "compute_") + sm;
        const char* opts[] = {arch.c_str(), "--std=c++14"};

        nvrtcProgram prog{};
        if (nvrtcCreateProgram(&prog, src.c_str(), "lodestone_solver.cu", 0, nullptr,
                               nullptr) != NVRTC_SUCCESS) {
            std::cerr << "nvrtcCreateProgram failed." << std::endl;
            return false;
        }

        const nvrtcResult cres = nvrtcCompileProgram(prog, 2, opts);
        if (cres != NVRTC_SUCCESS) {
            size_t logSize = 0;
            nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            if (logSize) nvrtcGetProgramLog(prog, &log[0]);
            nvrtcDestroyProgram(&prog);
            if (cubin) continue;  // retry as PTX before giving up
            std::cerr << "NVRTC failed to compile the kernel:\n" << log << std::endl;
            return false;
        }

        bool got = false;
        size_t size = 0;
        if (cubin) {
            if (nvrtcGetCUBINSize(prog, &size) == NVRTC_SUCCESS && size > 0) {
                image.assign(size, 0);
                got = nvrtcGetCUBIN(prog, image.data()) == NVRTC_SUCCESS;
            }
        } else {
            if (nvrtcGetPTXSize(prog, &size) == NVRTC_SUCCESS && size > 0) {
                image.assign(size, 0);
                got = nvrtcGetPTX(prog, image.data()) == NVRTC_SUCCESS;
            }
        }
        nvrtcDestroyProgram(&prog);
        if (got) return true;
        if (!cubin) break;
    }

    std::cerr << "Could not obtain compiled code for the kernel." << std::endl;
    return false;
}

static bool loadKernel(const std::string& src, const Gpu& gpu,
                       CUmodule& module, CUfunction& function) {
    std::vector<char> image;
    if (!compileKernel(src, gpu, image)) return false;
    CU_TRY(cuModuleLoadData(&module, image.data()));
    CU_TRY(cuModuleGetFunction(&function, module, "search_textures"));

    // Register count drives occupancy, which is the usual explanation when a
    // kernel that does strictly less arithmetic still runs slower.
    if (getenv("LODESTONE_DIAG")) {
        int regs = 0, localBytes = 0, maxThreads = 0;
        cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
        cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function);
        cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
        std::cerr << "[diag] " << regs << " regs, " << localBytes << " B local, max "
                  << maxThreads << " threads/block" << std::endl;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
static int envInt(const char* name, int fallback) {
    if (const char* v = getenv(name)) return atoi(v);
    return fallback;
}

// Lets the design knobs be swept without rebuilding. Defaults are the measured
// winner on this hardware class; see cuda_kernel_gen.h.
static KernelOptions kernelOptionsFromEnv() {
    KernelOptions o;
    o.fwdDiff = envInt("LODESTONE_FWDDIFF", o.fwdDiff ? 1 : 0) != 0;
    o.streams = envInt("LODESTONE_STREAMS", o.streams);
    o.bitmask = envInt("LODESTONE_BITMASK", o.bitmask ? 1 : 0) != 0;
    o.regBlocks = envInt("LODESTONE_REGBLOCKS", o.regBlocks);
    o.launchBounds = envInt("LODESTONE_LAUNCHBOUNDS", o.launchBounds);
    return o;
}

static unsigned envDim(const char* name, unsigned fallback) {
    if (const char* v = getenv(name)) {
        const int parsed = atoi(v);
        if (parsed > 0) return (unsigned)parsed;
    }
    return fallback;
}

// Slices the z axis and launches one grid per slice. This is not optional on
// Windows: WDDM's TDR watchdog resets the display driver if a single launch runs
// longer than ~2 seconds, so chunk length is adapted to stay far below that.
static bool runSearch(CUfunction function, Uniforms bounds,
                      int maxResults, int yOffsetFixup, bool showProgress,
                      std::vector<MatchResult>& outResults, int64_t& outCount,
                      double& outSeconds) {
    const int64_t xSpan = (int64_t)bounds.x_max - bounds.x_min + 1;
    const int64_t ySpan = (int64_t)bounds.y_max - bounds.y_min + 1;
    const int64_t zSpan = (int64_t)bounds.z_max - bounds.z_min + 1;

    CUdeviceptr dCount = 0, dResults = 0;
    CU_TRY(cuMemAlloc(&dCount, sizeof(int)));
    CU_TRY(cuMemsetD32(dCount, 0, 1));
    CU_TRY(cuMemAlloc(&dResults, (size_t)maxResults * sizeof(MatchResult)));

    // 128 threads/block. Swept on Ada: everything from 64 to 512 threads lands
    // within ~2% of this, but a single-warp block (32x1) is ~25% worse.
    const unsigned bx = envDim("LODESTONE_BX", 32);
    const unsigned by = envDim("LODESTONE_BY", 4);

    // Start small, then scale toward the target so the first launch cannot trip
    // TDR on a slow device and later launches are not needlessly short.
    const double kTargetSeconds = 0.25;
    int64_t rowsPerChunk = 1;
    {
        const int64_t perRow = (xSpan * ySpan) > 0 ? (xSpan * ySpan) : 1;
        rowsPerChunk = 200000000LL / perRow;
        if (rowsPerChunk < 1) rowsPerChunk = 1;
        if (rowsPerChunk > zSpan) rowsPerChunk = zSpan;
    }

    const auto start = std::chrono::high_resolution_clock::now();

    for (int64_t zStart = bounds.z_min; zStart <= bounds.z_max;) {
        const int64_t zEnd =
            (zStart + rowsPerChunk - 1 < bounds.z_max) ? zStart + rowsPerChunk - 1
                                                       : bounds.z_max;
        const int64_t zRows = zEnd - zStart + 1;

        Uniforms chunk = bounds;
        chunk.z_min = (int)zStart;
        chunk.z_max = (int)zEnd;

        void* args[] = {&chunk, &dCount, &dResults};

        const unsigned gx = (unsigned)((xSpan + bx - 1) / bx);
        const unsigned gy = (unsigned)((zRows + by - 1) / by);

        const auto chunkStart = std::chrono::high_resolution_clock::now();
        CU_TRY(cuLaunchKernel(function, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr));
        CU_TRY(cuCtxSynchronize());
        const double chunkSeconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                          chunkStart)
                .count();

        if (showProgress) {
            const double done = (double)(zEnd - bounds.z_min + 1) / (double)zSpan;
            std::cerr << "\rSearching... " << (int)(done * 100.0) << "%   " << std::flush;
        }

        zStart = zEnd + 1;

        // Adapt toward the target launch length, capped at 4x per step.
        if (chunkSeconds > 1e-6) {
            double scale = kTargetSeconds / chunkSeconds;
            if (scale > 4.0) scale = 4.0;
            if (scale < 0.25) scale = 0.25;
            int64_t next = (int64_t)((double)rowsPerChunk * scale);
            if (next < 1) next = 1;
            if (next > zSpan) next = zSpan;
            rowsPerChunk = next;
        }
    }

    if (showProgress) std::cerr << "\r                      \r" << std::flush;

    outSeconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start)
            .count();

    int count = 0;
    CU_TRY(cuMemcpyDtoH(&count, dCount, sizeof(int)));
    outCount = count;

    const int64_t stored = (outCount < maxResults) ? outCount : maxResults;
    outResults.assign((size_t)stored, MatchResult{0, 0, 0});
    if (stored > 0) {
        CU_TRY(cuMemcpyDtoH(outResults.data(), dResults,
                            (size_t)stored * sizeof(MatchResult)));
    }
    for (auto& r : outResults) r.y -= yOffsetFixup;

    cuMemFree(dCount);
    cuMemFree(dResults);
    return true;
}

static void printHelp(const char* prog) {
    std::cout
        << "Usage: " << prog << " [--rotate <direction>] [x_min x_max y_min y_max z_min z_max]\n\n"
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
        << "  LODESTONE_DEVICE   CUDA device index (default 0)\n"
        << "  LODESTONE_BX/BY    Thread block dimensions for tuning (default 64 x 4)\n\n"
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

static bool initGpu(Gpu& gpu) {
    CU_TRY(cuInit(0));

    int deviceCount = 0;
    CU_TRY(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found." << std::endl;
        return false;
    }

    int index = (int)envDim("LODESTONE_DEVICE", 0);
    if (getenv("LODESTONE_DEVICE") == nullptr) index = 0;
    if (index >= deviceCount) {
        std::cerr << "LODESTONE_DEVICE=" << index << " but only " << deviceCount
                  << " device(s) present." << std::endl;
        return false;
    }

    CU_TRY(cuDeviceGet(&gpu.device, index));
    CU_TRY(cuDeviceGetAttribute(&gpu.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                gpu.device));
    CU_TRY(cuDeviceGetAttribute(&gpu.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                gpu.device));

    char name[256] = {0};
    CU_TRY(cuDeviceGetName(name, sizeof(name), gpu.device));
    gpu.name = name;

    CU_TRY(cuCtxCreate(&gpu.context, 0, gpu.device));
    return true;
}

int main(int argc, char* argv[]) {
    CmdArgs args = parseArgs(argc, argv);
    if (args.showHelp) {
        printHelp(argv[0]);
        return 0;
    }

    Gpu gpu;
    if (!initGpu(gpu)) return 1;
    std::cout << "Using CUDA Device: " << gpu.name << " (sm_" << gpu.major << gpu.minor
              << ")" << std::endl;

    std::vector<RotationInfo> baseFormation = loadFormation();

    std::vector<RotateDir> dirsToSearch;
    if (args.rotate == RotateDir::UNKNOWN) {
        dirsToSearch = {RotateDir::NORTH, RotateDir::EAST, RotateDir::SOUTH, RotateDir::WEST};
    } else {
        dirsToSearch = {args.rotate};
    }

    int totalMatches = 0;
    double totalSeconds = 0;
    bool allOk = true;

    for (RotateDir dir : dirsToSearch) {
        if (args.rotate == RotateDir::UNKNOWN) {
            std::cout << "\n=== Searching Rotation: " << rotateDirName(dir) << " ===" << std::endl;
        }

        std::vector<RotationInfo> formation = rotateFormation(baseFormation, dir);
        orderByRejectionPower(formation);

        Uniforms uniforms;
        uniforms.x_min = args.x_min;
        uniforms.x_max = args.x_max;
        uniforms.y_min = args.y_min;
        uniforms.y_max = args.y_max;
        uniforms.z_min = args.z_min;
        uniforms.z_max = args.z_max;
        uniforms.num_blocks = (int)formation.size();

        if (!validateBounds(uniforms, formation)) continue;

        CUmodule module = nullptr;
        CUfunction function = nullptr;
        const std::string src =
            generateOptimizedKernel(formation, kDefaultMaxResults, kernelOptionsFromEnv());
        if (!loadKernel(src, gpu, module, function)) {
            allOk = false;
            break;
        }

        std::vector<MatchResult> results;
        int64_t count = 0;
        double seconds = 0;
        const bool ok =
            runSearch(function, uniforms, kDefaultMaxResults,
                      blockZeroYOffset(formation), true, results, count, seconds);

        cuModuleUnload(module);

        if (!ok) {
            allOk = false;
            break;
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
            std::cout << "\nCUDA GPU execution time: " << seconds << " seconds" << std::endl;
        }
    }

    if (args.rotate == RotateDir::UNKNOWN && allOk) {
        std::cout << "\nTotal execution time across all 4 rotations: " << totalSeconds << " seconds" << std::endl;
    }

    cuCtxDestroy(gpu.context);
    return allOk ? 0 : 1;
}

