#!/usr/bin/env python3
import sys
import os
import math
import argparse
import json

try:
    import nbtlib
except ImportError:
    print("Error: nbtlib is required. Install it using 'pip install nbtlib'")
    sys.exit(1)

FACING_MAP = {
    'north': 0,
    'east': 1,
    'south': 2,
    'west': 3,
}

def parse_litematic(file_path, target_block_name):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    nbt_file = nbtlib.load(file_path)
    regions = nbt_file.get('Regions', {})

    all_found_blocks = []
    target_clean = target_block_name.lower().replace("minecraft:", "")

    for region_name, region in regions.items():
        size_x = abs(int(region['Size']['x']))
        size_y = abs(int(region['Size']['y']))
        size_z = abs(int(region['Size']['z']))

        palette = region.get('BlockStatePalette', [])
        if not palette:
            continue

        bits_per_entry = max(2, math.ceil(math.log2(len(palette))))
        raw_states = region.get('BlockStates', [])
        block_states = [int(x) & 0xFFFFFFFFFFFFFFFF for x in raw_states]

        def get_palette_index(block_idx):
            bit_index = block_idx * bits_per_entry
            start_long = bit_index // 64
            start_offset = bit_index % 64

            val = block_states[start_long] >> start_offset
            if start_offset + bits_per_entry > 64:
                needed = (start_offset + bits_per_entry) - 64
                val |= (block_states[start_long + 1] & ((1 << needed) - 1)) << (bits_per_entry - needed)

            return val & ((1 << bits_per_entry) - 1)

        for y in range(size_y):
            for z in range(size_z):
                for x in range(size_x):
                    index = (y * size_z + z) * size_x + x
                    p_idx = get_palette_index(index)
                    block = palette[p_idx]
                    b_name = str(block.get('Name', '')).lower()
                    if target_clean in b_name:
                        props = block.get('Properties', {})
                        facing = str(props.get('facing', 'north')).lower()
                        rot_val = FACING_MAP.get(facing, 0)
                        all_found_blocks.append((x, y, z, b_name, facing, rot_val))

    return all_found_blocks


def generate_code(blocks, target_block_name, origin_index=0):
    if not blocks:
        print(f"No '{target_block_name}' blocks found in schematic.")
        return

    if origin_index >= len(blocks):
        origin_index = 0

    origin_x, origin_y, origin_z, _, _, _ = blocks[origin_index]
    print(f"// Total '{target_block_name}' blocks found: {len(blocks)}")
    print(f"// Selected origin block at index {origin_index}: ({origin_x}, {origin_y}, {origin_z})\n")

    # 1. Save universal formation.txt for Rust, Java, and Metal GPU solvers
    with open("formation.txt", "w") as f:
        f.write(f"{len(blocks)}\n")
        for x, y, z, _, _, rot in blocks:
            rel_x = x - origin_x
            rel_y = y - origin_y
            rel_z = z - origin_z
            f.write(f"{rel_x} {rel_y} {rel_z} {rot} 0\n")
    print("✅ Exported universal data to 'formation.txt'")

    # 2. Save formation.json
    json_data = {
        "block_count": len(blocks),
        "target_block": target_block_name,
        "blocks": []
    }
    for x, y, z, name, facing, rot in blocks:
        rel_x = x - origin_x
        rel_y = y - origin_y
        rel_z = z - origin_z
        json_data["blocks"].append({
            "x": rel_x,
            "y": rel_y,
            "z": rel_z,
            "rotation": rot,
            "is_side": False,
            "facing": facing,
            "block_name": name
        })

    with open("formation.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print("✅ Exported universal data to 'formation.json'\n")

    print("=== Rust Formation Code ===")
    print("fn default_formation() -> Vec<RotationInfo> {")
    print("    vec![")
    for x, y, z, name, facing, rot in blocks:
        rel_x = x - origin_x
        rel_y = y - origin_y
        rel_z = z - origin_z
        print(f"        RotationInfo::new({rel_x}, {rel_y}, {rel_z}, {rot}, false), // {name} (facing: {facing})")
    print("    ]")
    print("}\n")

    print("=== Java Formation Code ===")
    for x, y, z, name, facing, rot in blocks:
        rel_x = x - origin_x
        rel_y = y - origin_y
        rel_z = z - origin_z
        print(f"formation.add(new RotationInfo({rel_x}, {rel_y}, {rel_z}, {rot}, false)); // {name} (facing: {facing})")


def main():
    parser = argparse.ArgumentParser(description="Extract block orientations from a .litematic schematic for TextureRotations.")
    parser.add_argument("file", nargs="?", help="Path to .litematic file")
    parser.add_argument("-b", "--block", default="lime_glazed_terracotta", help="Block name to search for (default: lime_glazed_terracotta)")
    parser.add_argument("-o", "--origin-index", type=int, default=0, help="Index of the block to use as origin (default: 0)")

    args = parser.parse_args()

    print(f"Parsing litematica schematic: {args.file}")
    print(f"Target block: {args.block}")
    blocks = parse_litematic(args.file, args.block)
    generate_code(blocks, args.block, args.origin_index)


if __name__ == "__main__":
    main()
