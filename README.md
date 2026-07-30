# Lodestone

Lodestone is a high-performance, GPU-accelerated coordinate solver for Minecraft texture rotations. It searches world space $(x, y, z)$ to reverse-engineer exact structure or player coordinates by matching observed block texture rotation patterns against Minecraft's coordinate hashing algorithms (Vanilla & Sodium).

---

## Key Features

- **Multi-Platform GPU Engines**:
  - **Metal GPU Solver**: Optimized for Apple Silicon (M1/M2/M3/M4) using Objective-C++ and Metal compute pipelines.
  - **CUDA GPU Solver**: JIT kernel source generation tailored for NVIDIA GPUs and Windows environments.
- **Dynamic Shader Code Generation**: Unrolls search loops and embeds constant block offsets at compile-time to maximize GPU thread occupancy and instruction throughput.
- **Differential Verification Suite**: Integrated `--verify` command that validates GPU kernels against standard reference models across 12 automated test cases.
- **Schematic Parser**: Python utility (`parse_litematica.py`) for extracting block rotation matrices directly from `.litematic` schematics.
- **Interactive Visual Sampler**: Browser-based Canvas UI (`index.html`) featuring 2D perspective quad warping and homography grid overlays to measure block orientations from screenshots.

---

## Prerequisites & Setup

### macOS (Metal Solver)
- macOS 12+ (Apple Silicon or Intel Mac with Metal support)
- Xcode Command Line Tools (`xcode-select --install`) or Xcode IDE

### Windows (CUDA Solver)
- 64-bit Windows
- NVIDIA GPU with CUDA Toolkit (`nvcc`)
- Visual Studio C++ build tools (`cl.exe`)

### Python Parser
- Python 3.8+
- `nbtlib` dependency:
  ```bash
  pip install nbtlib
  ```

---

## Quick Start

### 1. Build the Solver

**macOS (Metal)**:
```bash
./build_metal.sh
```

**Windows (CUDA)**:
```cmd
build_cuda.bat
```

### 2. Run Verification Suite

Verify GPU kernel correctness and benchmark performance:
```bash
./metal_solver --verify
```

### 3. Generate Block Formation Data

#### Method A: From a `.litematic` Schematic
```bash
python3 parse_litematica.py path/to/schematic.litematic -b lime_glazed_terracotta
```
This generates `formation.txt` and `formation.json`.

#### Method B: Interactive Visual Sampler (Browser UI)
1. Open [index.html](file:///Users/lucaszhang/Lodestone/index.html) in your browser.
2. Load an in-game screenshot containing rotated blocks.
3. Draw quadrilaterals around target block faces to measure rotation angles using the homography grid overlay.

### 4. Run Coordinate Search

With `formation.txt` placed in the project root directory:
```bash
# Default search (assumes North orientation)
./metal_solver

# Search with explicit rotation direction (north, east, south, west)
./metal_solver --rotate east -200 200 60 128 -200 200

# Search all 4 cardinal rotations sequentially when orientation is unknown
./metal_solver --rotate unknown -6000 6000 60 256 -6000 6000
```


---

## Data Formats

### `formation.txt`
The solver reads candidate block formations from `formation.txt`:
```
<block_count>
<rel_x> <rel_y> <rel_z> <rotation> <is_side>
...
```

* `rel_x`, `rel_y`, `rel_z`: Relative block offset from the selected origin block.
* `rotation`: Texture rotation index (0–3 for top faces, 0–1 for side faces).
* `is_side`: `0` for mod-4 top face textures, `1` for mod-2 side face textures.

---

## Technical Details & Optimization

Minecraft computes block texture rotations by hashing world coordinates:

$$\text{hash} = \Big((x \times 3129871) \oplus (z \times 116129781)\Big) \oplus y$$
$$\text{seed} = (\text{hash} \times 42317861 + \text{hash} \times 11) \dots$$

Lodestone accelerates search throughput via:
1. **Early Rejection**: Orders candidate evaluation by rejection power (mod-4 blocks evaluated before mod-2 blocks), rejecting >75% of non-matching coordinate columns on the initial block check.
2. **Forward-Difference Y-Segments**: Batch-evaluates $Y$ ranges in 64-block segments to reduce global memory atomic bottlenecks.
3. **Register Reuse**: Stores column hashes for primary blocks in registers to minimize redundant math operations across vertical passes.
