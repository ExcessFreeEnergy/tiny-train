#!/usr/bin/env python3
"""
merge_dataset.py - Merge existing train_trimmed.bin with sampled Cosmopedia-v2 tokens to reach 1 Billion tokens.
"""

import os
import time

import numpy as np


def main():
    target_tokens = 1_000_000_000
    existing_bin = "data/TinyStories/train_trimmed.bin"
    output_bin = "data/TinyStories/train_1b.bin"
    cosmo_bins = [
        "data/cosmopedia_temp/cosmopedia_33.bin",
        "data/cosmopedia_temp/cosmopedia_34.bin",
    ]

    print("=== 1 Billion Token Dataset Merger ===")
    t0 = time.time()

    # Step 1: Inspect existing dataset
    if not os.path.exists(existing_bin):
        raise FileNotFoundError(f"Existing binary file '{existing_bin}' not found.")

    existing_data = np.memmap(existing_bin, dtype=np.uint16, mode="r")
    num_existing = len(existing_data)
    print(f"Existing dataset ('{existing_bin}'): {num_existing:,} tokens ({os.path.getsize(existing_bin) / (1024**2):.2f} MB)")

    needed_tokens = target_tokens - num_existing
    print(f"Target tokens: {target_tokens:,} | Needed additional tokens: {needed_tokens:,}")

    if needed_tokens <= 0:
        print("Dataset already contains >= 1B tokens. Nothing to merge.")
        return

    # Step 2: Load Cosmopedia token pools
    cosmo_arrays = []
    total_cosmo = 0
    for c_path in cosmo_bins:
        if os.path.exists(c_path):
            arr = np.memmap(c_path, dtype=np.uint16, mode="r")
            cosmo_arrays.append(arr)
            total_cosmo += len(arr)
            print(f"Loaded Cosmopedia pool '{c_path}': {len(arr):,} tokens")

    print(f"Total Cosmopedia tokens available: {total_cosmo:,}")
    if total_cosmo < needed_tokens:
        raise ValueError(f"Insufficient Cosmopedia tokens! Available: {total_cosmo:,}, Needed: {needed_tokens:,}")

    # Concatenate Cosmopedia pools (virtual or memmap view)
    cosmo_concat = np.concatenate(cosmo_arrays)

    # Step 3: Block-level random sampling to preserve document locality
    print(f"\nRandomly sampling {needed_tokens:,} tokens from Cosmopedia pools...")
    block_size = 2048
    num_blocks = len(cosmo_concat) // block_size
    usable_tokens = num_blocks * block_size
    cosmo_trimmed = cosmo_concat[:usable_tokens]

    blocks = cosmo_trimmed.reshape(num_blocks, block_size)
    rng = np.random.default_rng(seed=42)
    shuffled_indices = rng.permutation(num_blocks)

    blocks_needed = (needed_tokens + block_size - 1) // block_size
    selected_indices = shuffled_indices[:blocks_needed]
    sampled_blocks = blocks[selected_indices].reshape(-1)
    sampled_tokens = sampled_blocks[:needed_tokens]

    print(f"Sampled {len(sampled_tokens):,} tokens from {blocks_needed:,} shuffled blocks.")

    # Step 4: Write combined 1B dataset
    print(f"\nWriting 1 Billion token binary dataset to '{output_bin}'...")
    output_memmap = np.memmap(output_bin, dtype=np.uint16, mode="w+", shape=(target_tokens,))

    # Copy existing tokens
    print(f"Copying {num_existing:,} TinyStories tokens...")
    output_memmap[:num_existing] = existing_data

    # Copy sampled tokens
    print(f"Copying {len(sampled_tokens):,} Cosmopedia tokens...")
    output_memmap[num_existing : num_existing + len(sampled_tokens)] = sampled_tokens

    output_memmap.flush()
    del output_memmap

    output_size_bytes = os.path.getsize(output_bin)
    print(f"Successfully generated '{output_bin}'! File size: {output_size_bytes:,} bytes ({output_size_bytes / (1024**3):.3f} GB)")

    # Step 5: Replace train_trimmed.bin with 1B dataset
    print(f"Replacing '{existing_bin}' with '{output_bin}'...")
    os.replace(output_bin, existing_bin)

    print(f"\n✅ DATASET EXPANSION COMPLETE in {time.time() - t0:.2f}s!")
    print(f"Final Dataset Path: '{existing_bin}'")
    print(f"Total Tokens: {os.path.getsize(existing_bin) // 2:,} tokens")


if __name__ == "__main__":
    main()
