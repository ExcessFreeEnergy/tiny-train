#!/usr/bin/env python3
"""
retokenize.py - Fast dataset tokenization script using Gigatoken API with Vocabulary Trimming.

Features:
  - Uses Gigatoken native API (TextFileSource, JsonlFileSource, ParquetFileSource, Tokenizer.encode_files).
  - Verifies prerequisites (Rust/cargo, uv, submodule checkout) and compiles gigatoken automatically if needed.
  - Vocabulary Trimming: Detects and removes unused ("dead") tokens from the vocabulary, significantly reducing embedding matrix memory during training.
  - High performance: Tokenizes GBs of text per second.
  - Supports outputting to raw .bin (uint16/uint32 memory-mappable for tinygrad/NanoGPT), .npy, or .parquet.
  - Automatically handles document separators and EOS insertion.

Usage Examples:
  # Tokenize TinyStories train set with vocabulary trimming enabled
  python retokenize.py -i data/TinyStories/TinyStories-train.txt -o data/TinyStories/train.bin --trim-vocab --vocab-map-out data/TinyStories/vocab_map.json

  # Tokenize validation set using existing vocabulary map
  python retokenize.py -i data/TinyStories/TinyStories-valid.txt -o data/TinyStories/valid.bin --vocab-map-in data/TinyStories/vocab_map.json

  # Force recompilation of gigatoken from local submodule
  python retokenize.py -i data/TinyStories/TinyStories-valid.txt -o data/TinyStories/valid.bin --force-rebuild
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np


def ensure_gigatoken_installed(force_rebuild: bool = False, submodule_dir: str = "gigatoken"):
    """Verify prerequisites and compile/install gigatoken if not compiled or force_rebuild is True."""
    is_installed = False
    if not force_rebuild:
        try:
            import gigatoken as gt

            # Quick check to ensure rust backend gigatoken_rs is loaded & functional
            _ = gt.BPETokenizer
            is_installed = True
        except Exception:
            is_installed = False

    if is_installed and not force_rebuild:
        return

    print("Checking prerequisites for Gigatoken compilation...")
    # Check cargo / rustc
    cargo_bin = shutil.which("cargo") or shutil.which("rustc")
    if not cargo_bin:
        raise RuntimeError("Rust compiler (cargo/rustc) not found. Please install Rust (e.g. via https://rustup.rs).")

    # Check uv
    uv_bin = shutil.which("uv")
    if not uv_bin:
        raise RuntimeError("`uv` executable not found. Please install uv.")

    submodule_path = Path(submodule_dir)
    cargo_toml = submodule_path / "Cargo.toml"
    if not cargo_toml.exists():
        print(f"Submodule directory '{submodule_dir}' missing Cargo.toml. Initializing git submodule...")
        res = subprocess.run(["git", "submodule", "update", "--init", "--recursive", submodule_dir], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Warning: git submodule update output: {res.stderr}")

    if not cargo_toml.exists():
        raise RuntimeError(f"Could not find Cargo.toml in '{submodule_dir}'. Make sure the gigatoken submodule is cloned properly.")

    action = "Rebuilding" if force_rebuild else "Compiling"
    print(f"{action} and installing gigatoken from submodule '{submodule_dir}'...")
    build_cmd = [uv_bin, "add", f"./{submodule_dir}"]
    res = subprocess.run(build_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to uv pip install -e
        build_cmd_alt = [uv_bin, "pip", "install", "-e", f"./{submodule_dir}"]
        res_alt = subprocess.run(build_cmd_alt, capture_output=True, text=True)
        if res_alt.returncode != 0:
            raise RuntimeError(f"Failed to build gigatoken:\n{res.stderr}\n{res_alt.stderr}")

    print("Gigatoken compilation and installation successful.")


def get_eos_token_id(tokenizer, user_eos_id: int | None) -> int | None:
    """Determine the EOS token ID for the given tokenizer."""
    if user_eos_id is not None:
        return user_eos_id

    specials = tokenizer._special_tokens()
    # Common EOS token strings
    for candidate in ["<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>", "<|eos|>"]:
        if candidate in specials:
            return specials[candidate]

    # Fallback to backend special tokens or vocab lookups if available
    for tok_id, tok_bytes in tokenizer.vocab.items():
        if tok_bytes in [b"<|endoftext|>", b"<|end_of_text|>", b"</s>", b"<eos>", b"<|eos|>"]:
            return tok_id

    return None


def create_file_source(
    paths: list[str],
    file_type: str,
    separator: str | None,
    json_field: str,
    parquet_column: str,
    gt_module,
):
    """Create appropriate Gigatoken FileSource object."""
    if file_type == "text":
        sep_bytes = separator.encode("utf-8") if separator and separator.lower() != "none" else None
        return gt_module.TextFileSource(paths, separator=sep_bytes)
    elif file_type == "jsonl":
        return gt_module.JsonlFileSource(paths, field=json_field)
    elif file_type == "parquet":
        return gt_module.ParquetFileSource(paths, column=parquet_column)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def flatten_tokens_with_eos(tokens: ak.Array, eos_id: int | None, dtype: np.dtype) -> np.ndarray:
    """Flatten an awkward array of document tokens into a 1D numpy array, optionally inserting EOS."""
    doc_lens = ak.to_numpy(ak.num(tokens))
    num_docs = len(doc_lens)
    flat_raw = ak.to_numpy(ak.flatten(tokens))

    if eos_id is None:
        return flat_raw.astype(dtype, copy=False)

    total_tokens = np.sum(doc_lens, dtype=np.int64) + num_docs
    result = np.full(total_tokens, eos_id, dtype=dtype)

    # Compute output boundaries for vectorized copy
    ends = np.cumsum(doc_lens + 1)
    indices = np.ones(total_tokens, dtype=bool)
    indices[ends - 1] = False
    result[indices] = flat_raw.astype(dtype, copy=False)

    return result


def build_trimmed_vocab_map(
    flat_tokens: np.ndarray,
    orig_vocab_size: int,
    always_keep_ids: list[int] | None = None,
    min_count: int = 1,
) -> tuple[np.ndarray, dict[int, int]]:
    """Build a trimmed vocabulary mapping from a 1D array of token IDs.

    Returns:
        new_to_orig: 1D uint32 array mapping new token ID -> original token ID
        orig_to_new: dict mapping original token ID -> new token ID
    """
    max_token_id = int(flat_tokens.max()) if len(flat_tokens) > 0 else 0
    actual_vocab_size = max(orig_vocab_size, max_token_id + 1)

    counts = np.bincount(flat_tokens, minlength=actual_vocab_size)
    used_mask = counts >= min_count

    if always_keep_ids:
        for k_id in always_keep_ids:
            if k_id is not None and 0 <= k_id < actual_vocab_size:
                used_mask[k_id] = True

    new_to_orig = np.where(used_mask)[0].astype(np.uint32)
    orig_to_new = {int(orig_id): int(new_id) for new_id, orig_id in enumerate(new_to_orig)}
    return new_to_orig, orig_to_new


def remap_tokens_to_trimmed(
    flat_tokens: np.ndarray,
    orig_to_new: dict[int, int],
    orig_vocab_size: int,
    target_dtype: np.dtype,
    fallback_new_id: int = 0,
) -> np.ndarray:
    """Remap tokens from original token IDs to trimmed token IDs using a numpy lookup table."""
    max_token_id = int(flat_tokens.max()) if len(flat_tokens) > 0 else 0
    lut_size = max(orig_vocab_size, max_token_id + 1)

    lut = np.full(lut_size, fallback_new_id, dtype=target_dtype)
    for orig_id, new_id in orig_to_new.items():
        if orig_id < lut_size:
            lut[orig_id] = new_id

    return lut[flat_tokens]


def save_vocab_map(
    filepath: str,
    orig_to_new: dict[int, int],
    new_to_orig: np.ndarray,
    orig_vocab_size: int,
    min_count: int = 1,
):
    """Save vocabulary mapping metadata to JSON file."""
    data = {
        "original_vocab_size": orig_vocab_size,
        "trimmed_vocab_size": len(new_to_orig),
        "dead_tokens_removed": orig_vocab_size - len(new_to_orig),
        "reduction_percentage": round((1.0 - len(new_to_orig) / orig_vocab_size) * 100.0, 2),
        "min_count_threshold": min_count,
        "new_to_orig": [int(x) for x in new_to_orig],
        "orig_to_new": {str(k): int(v) for k, v in orig_to_new.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved vocabulary mapping to '{filepath}'")


def load_vocab_map(filepath: str) -> tuple[dict, dict[int, int], np.ndarray]:
    """Load vocabulary mapping metadata from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    new_to_orig = np.array(data["new_to_orig"], dtype=np.uint32)
    orig_to_new = {int(k): int(v) for k, v in data["orig_to_new"].items()}
    return data, orig_to_new, new_to_orig


def main():
    parser = argparse.ArgumentParser(
        description="Arbitrary dataset retokenization using Gigatoken API with Vocabulary Trimming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="Input path(s) or glob patterns (e.g. data/*.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output destination path (e.g. dataset.bin, dataset.npy, dataset.parquet)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="openai-community/gpt2",
        help="HuggingFace model ID or path to tokenizer",
    )
    parser.add_argument(
        "-f",
        "--file-type",
        choices=["auto", "text", "jsonl", "parquet"],
        default="auto",
        help="Input format (auto detects based on extension)",
    )
    parser.add_argument(
        "-s",
        "--separator",
        default="<|endoftext|>",
        help="Separator for plain text files (use 'none' for no split)",
    )
    parser.add_argument(
        "--json-field",
        default="text",
        help="Field name for JSONL files",
    )
    parser.add_argument(
        "--parquet-column",
        default="text",
        help="Column name for Parquet files",
    )
    parser.add_argument(
        "--add-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS token ID after each document when outputting flat binary/npy",
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        default=None,
        help="Override EOS token ID (auto-detected if omitted)",
    )
    parser.add_argument(
        "--trim-vocab",
        action="store_true",
        default=False,
        help="Perform vocabulary trimming to eliminate dead tokens",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=50,
        help="Minimum token occurrence frequency required in dataset to retain token in trimmed vocabulary (default: 50)",
    )
    parser.add_argument(
        "--vocab-map-out",
        type=str,
        default=None,
        help="Path to save vocabulary mapping JSON (defaults to <output_dir>/vocab_map.json if --trim-vocab is set)",
    )
    parser.add_argument(
        "--vocab-map-in",
        type=str,
        default=None,
        help="Path to existing vocabulary mapping JSON to apply",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "bin", "npy", "parquet"],
        default="auto",
        help="Output format (bin: raw uint16/uint32 bytes; npy: numpy array; parquet: awkward array)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "uint16", "uint32"],
        default="auto",
        help="Token ID integer type for bin/npy output (auto selects uint16 if vocab <= 65535)",
    )
    parser.add_argument(
        "--force-rebuild",
        "--rebuild-gigatoken",
        dest="force_rebuild",
        action="store_true",
        default=False,
        help="Force checking prerequisites and recompiling/reinstalling gigatoken from local submodule",
    )

    args = parser.parse_args()

    # Step 1: Ensure gigatoken is compiled & installed
    ensure_gigatoken_installed(force_rebuild=args.force_rebuild)
    import gigatoken as gt

    # Step 2: Resolve input paths
    resolved_paths = []
    for pattern in args.inputs:
        matched = glob.glob(pattern)
        if matched:
            resolved_paths.extend(matched)
        elif os.path.exists(pattern):
            resolved_paths.append(pattern)
        else:
            print(f"Warning: input pattern/path '{pattern}' did not match any files.", file=sys.stderr)

    if not resolved_paths:
        print("Error: No valid input files found.", file=sys.stderr)
        sys.exit(1)

    resolved_paths.sort()
    print(f"Input files ({len(resolved_paths)}):")
    total_input_bytes = 0
    for p in resolved_paths[:5]:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        total_input_bytes += os.path.getsize(p)
        print(f"  - {p} ({size_mb:.2f} MB)")
    if len(resolved_paths) > 5:
        print(f"  ... and {len(resolved_paths) - 5} more files")
        for p in resolved_paths[5:]:
            total_input_bytes += os.path.getsize(p)

    total_input_mb = total_input_bytes / (1024 * 1024)

    # Determine file type
    file_type = args.file_type
    if file_type == "auto":
        ext = Path(resolved_paths[0]).suffix.lower()
        if ext in [".jsonl", ".json"]:
            file_type = "jsonl"
        elif ext in [".parquet", ".pq"]:
            file_type = "parquet"
        else:
            file_type = "text"
    print(f"File type: {file_type}")

    # Load Tokenizer using Gigatoken API
    print(f"Loading tokenizer '{args.tokenizer}' via Gigatoken API...")
    t_start = time.time()
    tokenizer = gt.Tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    eos_id = get_eos_token_id(tokenizer, args.eos_token_id) if args.add_eos else None
    print(f"Tokenizer loaded in {time.time() - t_start:.2f}s | Vocab size: {vocab_size:,} | EOS Token ID: {eos_id}")

    # Create Gigatoken FileSource
    source = create_file_source(
        resolved_paths,
        file_type=file_type,
        separator=args.separator,
        json_field=args.json_field,
        parquet_column=args.parquet_column,
        gt_module=gt,
    )

    # Perform Tokenization
    print("Tokenizing using Gigatoken rust engine...")
    t_tok_start = time.time()
    tokens = tokenizer.encode_files(source)
    t_tok_end = time.time()
    tok_duration = t_tok_end - t_tok_start

    num_docs = len(tokens)
    raw_tok_count = int(ak.sum(ak.num(tokens)))
    mb_per_sec = total_input_mb / tok_duration if tok_duration > 0 else 0
    toks_per_sec = raw_tok_count / tok_duration if tok_duration > 0 else 0

    print(f"Tokenization Complete in {tok_duration:.3f}s!")
    print(f"  - Speed: {mb_per_sec:.2f} MB/s ({toks_per_sec / 1e6:.2f} Mtok/s)")
    print(f"  - Documents: {num_docs:,}")
    print(f"  - Tokens (raw): {raw_tok_count:,}")

    # Output Format Resolution
    out_format = args.format
    if out_format == "auto":
        out_ext = Path(args.output).suffix.lower()
        if out_ext == ".bin":
            out_format = "bin"
        elif out_ext == ".npy":
            out_format = "npy"
        elif out_ext in [".parquet", ".pq"]:
            out_format = "parquet"
        else:
            out_format = "bin"  # default to bin

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Flatten tokens
    flat_arr = flatten_tokens_with_eos(tokens, eos_id if args.add_eos else None, np.uint32)
    final_tok_count = len(flat_arr)

    # Vocabulary Trimming Handling
    final_vocab_size = vocab_size
    orig_to_new = None
    new_to_orig = None

    if args.vocab_map_in:
        print(f"Applying existing vocabulary map from '{args.vocab_map_in}'...")
        vocab_meta, orig_to_new, new_to_orig = load_vocab_map(args.vocab_map_in)
        final_vocab_size = len(new_to_orig)
        fallback_new_id = orig_to_new.get(eos_id, 0) if eos_id is not None else 0
        flat_arr = remap_tokens_to_trimmed(flat_arr, orig_to_new, vocab_size, np.uint32, fallback_new_id=fallback_new_id)
        print(f"  - Trimmed Vocab Size: {final_vocab_size:,} (Original: {vocab_size:,})")

    elif args.trim_vocab:
        print(f"Performing vocabulary trimming (min_count={args.min_count})...")
        t_trim_start = time.time()
        new_to_orig, orig_to_new = build_trimmed_vocab_map(
            flat_arr,
            orig_vocab_size=vocab_size,
            always_keep_ids=[eos_id] if eos_id is not None else None,
            min_count=args.min_count,
        )
        final_vocab_size = len(new_to_orig)
        dead_tokens = vocab_size - final_vocab_size
        reduction_pct = (dead_tokens / vocab_size) * 100.0
        fallback_new_id = orig_to_new.get(eos_id, 0) if eos_id is not None else 0
        flat_arr = remap_tokens_to_trimmed(flat_arr, orig_to_new, vocab_size, np.uint32, fallback_new_id=fallback_new_id)
        t_trim_end = time.time()
        print(f"Vocabulary Trimming Summary (in {t_trim_end - t_trim_start:.3f}s):")
        print(f"  - Original Vocab Size: {vocab_size:,}")
        print(f"  - Active (Used) Tokens: {final_vocab_size:,}")
        print(f"  - Min Count Threshold: {args.min_count}")
        print(f"  - Dead/Rare Tokens Removed: {dead_tokens:,} ({reduction_pct:.2f}%)")
        print(f"  - Embedding Matrix Size: ({final_vocab_size:,}, d_model)")

        map_out_path = args.vocab_map_out or os.path.join(os.path.dirname(os.path.abspath(args.output)), "vocab_map.json")
        save_vocab_map(map_out_path, orig_to_new, new_to_orig, orig_vocab_size=vocab_size, min_count=args.min_count)

    # Target dtype
    if args.dtype == "uint16" or (args.dtype == "auto" and final_vocab_size <= 65535):
        target_dtype = np.uint16
    else:
        target_dtype = np.uint32

    flat_arr = flat_arr.astype(target_dtype, copy=False)

    print(f"Exporting to '{args.output}' (format: {out_format}, dtype: {target_dtype.__name__})...")
    t_save_start = time.time()

    if out_format == "parquet":
        # Save awkward array directly (unflatten remapped flat array if trimming applied)
        if orig_to_new is not None:
            export_tokens = ak.unflatten(flat_arr, ak.num(tokens))
        else:
            export_tokens = tokens
        ak.to_parquet(export_tokens, args.output)
    elif out_format == "bin":
        flat_arr.tofile(args.output)
    elif out_format == "npy":
        np.save(args.output, flat_arr)

    t_save_end = time.time()
    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Export completed in {t_save_end - t_save_start:.3f}s!")
    print(f"  - Final output path: {args.output}")
    print(f"  - Output size: {out_size_mb:.2f} MB")
    print(f"  - Total token count: {final_tok_count:,}")

    # Decoded sanity check
    if num_docs > 0:
        sample_doc = tokens[0]
        sample_text = tokenizer.decode(sample_doc)
        print("\n--- Decoded Sample Check (Doc 1) ---")
        preview = sample_text[:200].decode("utf-8", errors="replace")
        print(f"Original Tokens (first 10): {sample_doc[:10]}")
        if orig_to_new is not None:
            trimmed_sample = [orig_to_new[int(tok)] for tok in sample_doc[:10]]
            print(f"Trimmed Tokens (first 10):  {trimmed_sample}")
            # Verify inverse mapping decodes back perfectly
            recovered = [int(new_to_orig[tok]) for tok in trimmed_sample]
            decoded_back = tokenizer.decode(recovered).decode("utf-8", errors="replace")
            print(f"Decoded back from trimmed:  {decoded_back[:100]!r}...")
        else:
            print(f"Text preview: {preview!r}...")
        print("------------------------------------\n")


if __name__ == "__main__":
    main()
