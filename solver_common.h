// Pieces shared by metal_solver.mm and cuda_solver.cpp.
//
// Deliberately free of Metal and CUDA types so both toolchains can compile it.
// Anything platform-specific (pipeline creation, dispatch, memory) stays in the
// per-backend file; what lives here is the formation format, the search bounds
// contract, and the shared constants, so the two backends cannot drift apart on
// any of them.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// One observed block: position relative to the formation origin, its texture
// rotation, and whether it is a side face (mod 2) rather than a top face (mod 4).
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

// Vanilla coordinate hash constants.
static const int32_t kMulX = 3129871;
static const int64_t kMulZ = 116129781LL;
static const int64_t kHashA = 42317861LL;

// Number of per-block column hashes kept in registers. Blocks past this index are
// reached with probability <= 4^-kRegBlocks, so recomputing them inline (2 adds +
// an xor) is cheaper than the occupancy cost of holding them. Swept on an M1;
// re-sweep per backend.
static const int kRegBlocks = 4;

// Width of a forward-difference y segment. Fixed at 64 because each segment's
// block-0 results are accumulated into a single 64-bit mask. Only full segments
// take the fast path; 64 also keeps waste low on a typical 200-300 tall search,
// where a wider segment would push most of the range onto the direct path.
static const int kSegment = 64;

static const int kDefaultMaxResults = 4096;

// The y recovered inside the kernel is offset by block 0's b.y; undo it on readback.
inline int blockZeroYOffset(const std::vector<RotationInfo>& f) { return f[0].y; }

// ---------------------------------------------------------------------------
// formation.txt: "<count>\n" then one "<x> <y> <z> <rotation> <is_side>" per line.
// Written by parse_litematica.py; read identically by every backend.
// ---------------------------------------------------------------------------
inline std::vector<RotationInfo> loadFormation() {
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
inline void orderByRejectionPower(std::vector<RotationInfo>& f) {
    std::stable_partition(f.begin(), f.end(),
                          [](const RotationInfo& b) { return b.is_side == 0; });
}

// The kernel folds (z + b.z) into pz + b.z*kMulZ, which agrees with the vanilla
// hash unless (z + b.z) overflows int32. Reject those bounds.
inline bool validateBounds(const Uniforms& u, const std::vector<RotationInfo>& f) {
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
