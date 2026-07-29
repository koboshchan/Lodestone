#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>

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

const char* metal_shader_src = R"(
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

inline long get_coord_random_fast(long part_xz, int y) {
    long l = part_xz ^ (long)y;
    return l * l * 42317861L + l * 11L;
}

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

    // Precompute constant X & Z hash parts for this column
    long part_x = (long)(x * 3129871);
    long part_z = (long)z * 116129781L;
    long part_xz = part_x ^ part_z;

    int num_blocks = uniforms.num_blocks;

    // Cache first 4 blocks in thread local registers for fast early rejection
    RotationInfo b0 = blocks[0];
    int mod0 = b0.is_side ? 2 : 4;

    for (int y = uniforms.y_min; y <= uniforms.y_max; y++) {
        // Fast-path test on origin block
        long l0 = (long)((x + b0.x) * 3129871) ^ ((long)(z + b0.z) * 116129781L);
        if (b0.rotation != get_texture_fast(l0, x + b0.x, y + b0.y, z + b0.z, mod0)) {
            continue; // Rejects ~75% instantly
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
            if (idx < 100) {
                results[idx] = MatchResult{x, y, z};
            }
        }
    }
}
)";

std::vector<RotationInfo> loadFormation() {
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

    int count;
    file >> count;
    std::vector<RotationInfo> formation(count);
    for (int i = 0; i < count; i++) {
        file >> formation[i].x >> formation[i].y >> formation[i].z >> formation[i].rotation >> formation[i].is_side;
    }
    std::cout << "Loaded " << count << " block observations from formation.txt" << std::endl;
    return formation;
}

void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [x_min x_max y_min y_max z_min z_max]\n\n"
              << "Arguments:\n"
              << "  x_min x_max  Search bounds for X coordinates (default: -6000 6000)\n"
              << "  y_min y_max  Search bounds for Y coordinates (default: 60 256)\n"
              << "  z_min z_max  Search bounds for Z coordinates (default: -6000 6000)\n\n"
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

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:metal_shader_src];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            std::cerr << "Failed to compile Metal shader: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        id<MTLFunction> searchFunc = [library newFunctionWithName:@"search_textures"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:searchFunc error:&error];
        if (!pipeline) {
            std::cerr << "Failed to create pipeline state: " << [[error localizedDescription] UTF8String] << std::endl;
            return 1;
        }

        std::vector<RotationInfo> formation = loadFormation();

        int x_min = (argc > 1) ? atoi(argv[1]) : -6000;
        int x_max = (argc > 2) ? atoi(argv[2]) : 6000;
        int y_min = (argc > 3) ? atoi(argv[3]) : 60;
        int y_max = (argc > 4) ? atoi(argv[4]) : 256;
        int z_min = (argc > 5) ? atoi(argv[5]) : -6000;
        int z_max = (argc > 6) ? atoi(argv[6]) : 6000;

        Uniforms uniforms = {
            .x_min = x_min, .x_max = x_max,
            .y_min = y_min, .y_max = y_max,
            .z_min = z_min, .z_max = z_max,
            .num_blocks = (int)formation.size()
        };

        id<MTLBuffer> blocksBuffer = [device newBufferWithBytes:formation.data()
                                                         length:formation.size() * sizeof(RotationInfo)
                                                        options:MTLResourceStorageModeShared];

        id<MTLBuffer> uniformsBuffer = [device newBufferWithBytes:&uniforms
                                                          length:sizeof(Uniforms)
                                                         options:MTLResourceStorageModeShared];

        int zero = 0;
        id<MTLBuffer> countBuffer = [device newBufferWithBytes:&zero
                                                        length:sizeof(int)
                                                       options:MTLResourceStorageModeShared];

        std::vector<MatchResult> results(100);
        id<MTLBuffer> resultsBuffer = [device newBufferWithBytes:results.data()
                                                          length:results.size() * sizeof(MatchResult)
                                                         options:MTLResourceStorageModeShared];

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:blocksBuffer offset:0 atIndex:0];
        [encoder setBuffer:uniformsBuffer offset:0 atIndex:1];
        [encoder setBuffer:countBuffer offset:0 atIndex:2];
        [encoder setBuffer:resultsBuffer offset:0 atIndex:3];

        NSUInteger gridWidth = (uniforms.x_max - uniforms.x_min + 1);
        NSUInteger gridHeight = (uniforms.z_max - uniforms.z_min + 1);

        MTLSize gridSize = MTLSizeMake(gridWidth, gridHeight, 1);

        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

        auto start = std::chrono::high_resolution_clock::now();

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        int count = *(int*)[countBuffer contents];
        MatchResult* matchPtr = (MatchResult*)[resultsBuffer contents];

        std::cout << "\nFound " << count << " match(es):" << std::endl;
        for (int i = 0; i < count && i < 100; i++) {
            std::cout << "X: " << matchPtr[i].x << " Y: " << matchPtr[i].y << " Z: " << matchPtr[i].z << std::endl;
        }

        std::cout << "\nMetal GPU execution time: " << diff.count() << " seconds" << std::endl;
    }
    return 0;
}
