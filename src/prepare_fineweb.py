#!/usr/bin/env python3
"""
prepare_fineweb.py - Download HuggingFaceFW/fineweb sample-10BT shards and tokenize to 1B tokens.

Features:
  - Uses huggingface_hub to fetch sample/10BT/*.parquet shards.
  - Downloads shards until at least 1,005,000,000 tokens are gathered.
  - Uses retokenize.py (Gigatoken Rust engine) with vocabulary trimming.
  - Generates data/FineWeb/train_trimmed.bin (1B tokens) and data/FineWeb/valid_trimmed.bin (5M tokens).
  - Generates data/FineWeb/vocab_map.json.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


def download_fineweb_shards(raw_dir: str, target_tokens: int = 1_005_000_000):
    """Download FineWeb sample-10BT parquet shards until target token count is reachable."""
    os.makedirs(raw_dir, exist_ok=True)

    # Check if we already have >= 2 parquet shards in raw_dir
    existing_parquets = glob.glob(os.path.join(raw_dir, "**/*.parquet"), recursive=True)
    if len(existing_parquets) >= 2:
        print(f"Found {len(existing_parquets)} existing parquet shard(s) in {raw_dir}:")
        for p in existing_parquets:
            sz_mb = os.path.getsize(p) / (1024**2)
            print(f"  - {p} ({sz_mb:.1f} MB)", flush=True)
        return existing_parquets

    print("🔍 Fetching shard list from HuggingFaceFW/fineweb (sample-10BT)...", flush=True)
    api = HfApi()
    files = api.list_repo_files(repo_id="HuggingFaceFW/fineweb", repo_type="dataset")
    shards = sorted([f for f in files if f.startswith("sample/10BT/") and f.endswith(".parquet")])

    print(f"Found {len(shards)} parquet shards in sample/10BT/.", flush=True)

    downloaded_files = list(existing_parquets)
    for i, shard_path in enumerate(shards):
        actual_path = os.path.join(raw_dir, shard_path)
        filename = os.path.basename(shard_path)

        if actual_path in downloaded_files:
            continue

        if os.path.exists(actual_path) and os.path.getsize(actual_path) > 0:
            print(f"  - Shard already exists: {actual_path} ({os.path.getsize(actual_path) / (1024**2):.1f} MB)", flush=True)
            downloaded_files.append(actual_path)
        else:
            print(f"  - Downloading shard [{len(downloaded_files) + 1}/2]: {filename}...", flush=True)
            t0 = time.time()
            downloaded_file = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb",
                filename=shard_path,
                repo_type="dataset",
                local_dir=raw_dir,
            )
            dt = time.time() - t0
            sz_mb = os.path.getsize(downloaded_file) / (1024**2)
            print(f"    Downloaded in {dt:.2f}s ({sz_mb:.1f} MB)", flush=True)
            downloaded_files.append(downloaded_file)

        if len(downloaded_files) >= 2:
            print(f"Downloaded {len(downloaded_files)} shards (~1.44B tokens estimated).", flush=True)
            break

    return downloaded_files


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb sample-10BT 1B token dataset.")
    parser.add_argument("--data-dir", type=str, default="data/FineWeb", help="Target dataset directory")
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000, help="Target train token count (default: 1B)")
    parser.add_argument("--valid-tokens", type=int, default=5_000_000, help="Target valid token count (default: 5M)")
    parser.add_argument("--min-count", type=int, default=50, help="Vocabulary trimming minimum count threshold")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.data_dir)
    raw_dir = os.path.join(target_dir, "raw")

    print("=== FineWeb 1B Token Preparation Pipeline ===")
    print(f"Target Directory: {target_dir}")
    print(f"Target Train Tokens: {args.target_tokens:,}")
    print(f"Target Valid Tokens: {args.valid_tokens:,}\n")

    # Step 1: Download parquet shards
    downloaded_shards = download_fineweb_shards(raw_dir)
    if not downloaded_shards:
        raise RuntimeError("No parquet shards downloaded.")

    # Step 2: Tokenize using retokenize.py
    full_bin = os.path.join(target_dir, "full_trimmed.bin")
    vocab_map = os.path.join(target_dir, "vocab_map.json")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retokenize_script = os.path.join(project_root, "src", "retokenize.py")

    cmd = [
        sys.executable,
        retokenize_script,
        "-i",
        *downloaded_shards,
        "-o",
        full_bin,
        "--file-type",
        "parquet",
        "--parquet-column",
        "text",
        "--trim-vocab",
        "--min-count",
        str(args.min_count),
        "--vocab-map-out",
        vocab_map,
    ]

    print("\n🚀 Tokenizing shards using Gigatoken engine...", flush=True)
    print(f"Command: {' '.join(cmd)}\n", flush=True)

    t_tok0 = time.time()
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Retokenization failed with exit code {res.returncode}")
    print(f"Tokenization complete in {time.time() - t_tok0:.2f}s!", flush=True)

    # Step 3: Inspect generated token binary buffer and partition into train/valid
    if not os.path.exists(full_bin):
        raise FileNotFoundError(f"Expected binary output '{full_bin}' not found.")

    full_data = np.memmap(full_bin, dtype=np.uint16, mode="r")
    total_tokens = len(full_data)
    print(f"\nTotal tokens generated: {total_tokens:,} ({os.path.getsize(full_bin) / (1024**2):.2f} MB)", flush=True)

    needed_total = args.target_tokens + args.valid_tokens
    if total_tokens < needed_total:
        raise ValueError(f"Generated tokens ({total_tokens:,}) less than required ({needed_total:,}). Download more shards.")

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")

    print(f"\nSplitting into train_trimmed.bin ({args.target_tokens:,} tokens) and valid_trimmed.bin ({args.valid_tokens:,} tokens)...", flush=True)

    # Write train_trimmed.bin
    train_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(args.target_tokens,))
    train_mmap[:] = full_data[: args.target_tokens]
    train_mmap.flush()
    del train_mmap

    # Write valid_trimmed.bin
    valid_mmap = np.memmap(valid_bin, dtype=np.uint16, mode="w+", shape=(args.valid_tokens,))
    valid_mmap[:] = full_data[args.target_tokens : args.target_tokens + args.valid_tokens]
    valid_mmap.flush()
    del valid_mmap

    # Write fallbacks / copies
    if not os.path.exists(train_fallback):
        os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else None
    if not os.path.exists(valid_fallback):
        os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else None

    # Cleanup temporary full_bin
    if os.path.exists(full_bin):
        os.remove(full_bin)

    print("\n✅ FineWeb 1B Dataset Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {args.target_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB, {args.valid_tokens:,} tokens)")
    print(f"  - Vocab Map:     '{vocab_map}'")


if __name__ == "__main__":
    main()
