#!/usr/bin/env bash
# Builds the Metal texture-rotation solver to ./metal_solver.
#
# The bare `clang++` on this machine defaults to an SDK path that may not exist
# (e.g. CommandLineTools/SDKs/MacOSX27.sdk), which fails with
# "'Foundation/Foundation.h' file not found". So resolve a working SDK explicitly
# rather than trusting the compiler default.
#
# Usage:
#   ./build_metal.sh                 # build
#   ./build_metal.sh -g -O0          # extra flags are passed to the compiler
#   SDKROOT=/path/to/MacOSX.sdk ./build_metal.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/metal_solver.mm"
OUT="$ROOT/metal_solver"

# An SDK is only usable if it actually carries the frameworks we link against.
sdk_is_valid() {
    [ -n "${1:-}" ] && [ -d "$1/System/Library/Frameworks/Foundation.framework" ] \
                    && [ -d "$1/System/Library/Frameworks/Metal.framework" ]
}

resolve_sdk() {
    local candidate
    # Explicit override wins.
    if [ -n "${SDKROOT:-}" ]; then
        sdk_is_valid "$SDKROOT" && { printf '%s\n' "$SDKROOT"; return 0; }
        echo "warning: SDKROOT=$SDKROOT is not a usable SDK, ignoring" >&2
    fi
    # Whatever the active developer directory points at. SDKROOT is cleared for
    # this call: xcrun honours it, so a bogus value would poison its answer too.
    if candidate="$(env -u SDKROOT xcrun --sdk macosx --show-sdk-path 2>/dev/null)"; then
        sdk_is_valid "$candidate" && { printf '%s\n' "$candidate"; return 0; }
    fi
    # Known install locations, unversioned symlink first then any versioned SDK.
    local dir
    for dir in \
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs \
        /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs \
        /Library/Developer/CommandLineTools/SDKs
    do
        [ -d "$dir" ] || continue
        for candidate in "$dir/MacOSX.sdk" "$dir"/MacOSX*.sdk; do
            sdk_is_valid "$candidate" && { printf '%s\n' "$candidate"; return 0; }
        done
    done
    return 1
}

if ! SDK="$(resolve_sdk)"; then
    echo "error: no usable macOS SDK found." >&2
    echo "       Install Xcode or the Command Line Tools (xcode-select --install)," >&2
    echo "       or point SDKROOT at an SDK containing Metal.framework." >&2
    exit 1
fi

# Prefer Xcode's clang++ over anything earlier on PATH (a Homebrew LLVM clang++
# will not reliably build Objective-C++ against the system frameworks).
CXX="$(env -u SDKROOT xcrun -f clang++ 2>/dev/null || command -v clang++)"

echo "SDK:      $SDK"
echo "Compiler: $CXX"

"$CXX" -O3 -std=c++17 -fobjc-arc \
    -isysroot "$SDK" \
    -framework Metal -framework Foundation \
    "$SRC" -o "$OUT" "$@"

echo "Built:    $OUT"
