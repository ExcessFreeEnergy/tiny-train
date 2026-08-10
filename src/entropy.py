#!/usr/bin/env python3
"""entropy.py - Approximates dataset information density to recommend optimal model sizes."""

import argparse
import math
import os
import zlib
from collections import Counter


def calculate_shannon_entropy(data: bytes) -> float:
    """Calculates byte-level Shannon entropy: H = -sum(p * log2(p))."""
    if not data:
        return 0.0

    counts = Counter(data)
    length = len(data)
    entropy = 0.0

    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def estimate_model_architecture(target_params: int):
    """
    Derives layers, heads, and d_model based on standard Transformer scaling.

    Approximates Params = 12 * L * d_model^2, keeping L proportional to d_model.
    """
    # Clamp to reasonable minimums
    target_params = max(1_000_000, target_params)

    # Heuristic: d^3 = target_params * (64 / 12) assuming L = d / 64
    d_model_raw = (target_params * (64 / 12)) ** (1 / 3)

    # Snap d_model to nearest multiple of 64 for Tensor Core alignment
    d_model = max(64, int(round(d_model_raw / 64) * 64))

    # Heads: assume head_dim = 64
    n_heads = d_model // 64

    # Recalculate layers needed to hit target parameters
    # L = Params / (12 * d^2)
    n_layers = max(2, int(round(target_params / (12 * (d_model**2)))))

    # Recalculate actual params of this architecture
    actual_params = 12 * n_layers * (d_model**2)

    return n_layers, n_heads, d_model, actual_params


def main():
    parser = argparse.ArgumentParser(description="Estimate optimal model size from dataset entropy.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset (.bin, .txt, etc.)")
    parser.add_argument("--sample-mb", type=int, default=10, help="Megabytes to sample (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: File not found at {args.dataset_path}")
        return

    sample_bytes = args.sample_mb * 1024 * 1024

    print(f"Analyzing up to {args.sample_mb} MB of: {args.dataset_path}...")

    with open(args.dataset_path, "rb") as f:
        data = f.read(sample_bytes)

    if not data:
        print("File is empty.")
        return

    # 1. Byte-level Shannon Entropy
    # Max entropy for 256 byte values is log2(256) = 8.0
    shannon = calculate_shannon_entropy(data)

    # 2. Sequence-level Entropy Approximation (Lempel-Ziv)
    # Compression ratio = compressed size / original size
    compressed_data = zlib.compress(data, level=9)
    compression_ratio = len(compressed_data) / len(data)

    # 3. Model Sizing Heuristic
    # Assume a standard corpus (e.g., OpenWebText) has a compression ratio of ~0.35
    # and maps well to a 100M parameter model.
    # We use an exponential scale because sequence dependencies scale non-linearly.
    base_ratio = 0.35
    base_params = 100_000_000

    scale_factor = (compression_ratio / base_ratio) ** 2.5
    recommended_params = int(base_params * scale_factor)

    layers, heads, d_model, actual_params = estimate_model_architecture(recommended_params)

    print("\n" + "=" * 50)
    print("📊 DATASET ENTROPY PROFILE")
    print("=" * 50)
    print(f"Shannon Entropy (Byte-level): {shannon:.3f} / 8.000 bits")
    print(f"LZ Compression Ratio:         {compression_ratio:.3f} (Lower = highly predictable)")

    print("\n" + "=" * 50)
    print("🧠 RECOMMENDED TRANSFORMER ARCHITECTURE")
    print("=" * 50)
    if compression_ratio < 0.25:
        print("Classification: Low Entropy (e.g., TinyStories, Simple Logs)")
    elif compression_ratio < 0.45:
        print("Classification: Medium Entropy (e.g., Wikipedia, Novels)")
    else:
        print("Classification: High Entropy (e.g., Math, Source Code, Hex Dumps)")

    print(f"\nTarget Capacity:  ~{recommended_params:,} parameters")
    print(f"Snapped Capacity: ~{actual_params:,} parameters")
    print(f"Layers (L):       {layers}")
    print(f"Heads (H):        {heads}")
    print(f"Dimension (D):    {d_model}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
