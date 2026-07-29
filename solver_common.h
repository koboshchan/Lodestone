// Pieces shared by metal_solver.mm and cuda_solver.cpp.
//
// Deliberately free of Metal and CUDA types so both toolchains can compile it.
// Anything platform-specific (pipeline creation, dispatch, memory) stays in the
// per-backend file; what lives here is the formation format, the search bounds
// contract, and the differential-verification case list, so the two backends
// cannot drift apart on any of them.
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

inline bool operator<(const MatchResult& a, const MatchResult& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}
inline bool operator==(const MatchResult& a, const MatchResult& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

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

// The optimized kernel folds (z + b.z) into pz + b.z*kMulZ, which differs from
// the reference only if (z + b.z) overflows int32. Reject those bounds.
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

// ---------------------------------------------------------------------------
// Differential verification: shared case list and reporting
// ---------------------------------------------------------------------------
struct VerifyCase {
    const char* name;
    std::vector<RotationInfo> formation;
    Uniforms bounds;
};

// 4.2M entries, 50 MB. Cases are sized so neither kernel overflows this.
static const int kVerifyMaxResults = 1 << 22;

// 1- and 2-block formations match ~25% and ~6% of all coordinates, so these
// compare the hash at effectively every point in the box. A rare 20-block
// formation on its own would only ever prove empty == empty.
inline std::vector<VerifyCase> makeVerifyCases() {
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

    std::vector<RotationInfo> real = loadFormation();

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
    for (auto& c : cases) c.bounds.num_blocks = (int)c.formation.size();
    return cases;
}

struct VerifyTally {
    int failures = 0;
    int vacuous = 0;
};

// Sorts both result sets, compares them, prints the verdict, and updates the tally.
// Returns true if the case agreed.
inline bool reportVerifyCase(const char* name,
                             std::vector<MatchResult>& refOut, int64_t refCount,
                             std::vector<MatchResult>& optOut, int64_t optCount,
                             double refSecs, double optSecs,
                             int maxResults, VerifyTally& tally) {
    std::sort(refOut.begin(), refOut.end());
    std::sort(optOut.begin(), optOut.end());

    bool overflow = refCount > maxResults || optCount > maxResults;
    bool ok = !overflow && refCount == optCount && refOut == optOut;

    // Zero matches on both kernels agrees trivially and proves nothing; say so
    // rather than reporting a pass that carries no information.
    const char* verdict = !ok ? "  FAIL  " : (refCount == 0 ? "  VOID  " : "  PASS  ");
    if (ok && refCount == 0) tally.vacuous++;

    std::cout << verdict << name << "  (" << refCount << " matches";
    if (refSecs > 0 && optSecs > 0) std::cout << ", " << (refSecs / optSecs) << "x";
    std::cout << ")" << std::endl;

    if (!ok) {
        tally.failures++;
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
    return ok;
}

inline int reportVerifySummary(const VerifyTally& tally) {
    std::cout << std::endl;
    if (tally.failures == 0) {
        std::cout << "All verification cases passed." << std::endl;
    } else {
        std::cout << tally.failures << " case(s) FAILED." << std::endl;
    }
    if (tally.vacuous > 0) {
        std::cout << tally.vacuous << " case(s) marked VOID: both kernels found nothing, "
                     "so they agree only trivially." << std::endl;
    }
    return tally.failures == 0 ? 0 : 1;
}
