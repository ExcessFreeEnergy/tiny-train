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
import gc
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


def stream_text_batches(
    filepath: str,
    file_type: str,
    json_field: str,
    parquet_column: str,
    separator: str | None,
    batch_size: int = 5000,
):
    """Yield lists of document text strings in batches from a file."""
    if file_type == "parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filepath)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[parquet_column]):
            raw_list = batch.column(0).to_pylist()
            text_list = [t if (t is not None and isinstance(t, str)) else "" for t in raw_list]
            if text_list:
                yield text_list
    elif file_type == "jsonl":
        current_batch = []
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get(json_field, "")
                    if text:
                        current_batch.append(text)
                except Exception:
                    pass
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
            if current_batch:
                yield current_batch
    else:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            if separator and separator.lower() != "none":
                content = f.read()
                docs = content.split(separator)
                for i in range(0, len(docs), batch_size):
                    chunk = [d.strip() for d in docs[i : i + batch_size] if d.strip()]
                    if chunk:
                        yield chunk
            else:
                current_batch = []
                for line in f:
                    line = line.strip()
                    if line:
                        current_batch.append(line)
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
                if current_batch:
                    yield current_batch


def build_trimmed_vocab_map_from_counts(
    counts: np.ndarray,
    orig_vocab_size: int,
    always_keep_ids: list[int] | None = None,
    min_count: int = 1,
) -> tuple[np.ndarray, dict[int, int]]:
    """Build a trimmed vocabulary mapping from token frequency counts.

    Returns:
        new_to_orig: 1D uint32 array mapping new token ID -> original token ID
        orig_to_new: dict mapping original token ID -> new token ID
    """
    actual_vocab_size = max(orig_vocab_size, len(counts))
    if len(counts) < actual_vocab_size:
        counts = np.pad(counts, (0, actual_vocab_size - len(counts)))

    used_mask = counts >= min_count

    if always_keep_ids:
        for k_id in always_keep_ids:
            if k_id is not None and 0 <= k_id < actual_vocab_size:
                used_mask[k_id] = True

    new_to_orig = np.where(used_mask)[0].astype(np.uint32)
    orig_to_new = {int(orig_id): int(new_id) for new_id, orig_id in enumerate(new_to_orig)}
    return new_to_orig, orig_to_new


def flatten_and_remap_batch(
    flat_raw: np.ndarray,
    doc_lens: np.ndarray,
    eos_id: int | None,
    lut: np.ndarray | None,
    target_dtype: np.dtype,
    new_eos_id: int = 0,
) -> np.ndarray:
    """Remap tokens and insert EOS token ID per document into a 1D numpy array of target_dtype."""
    if len(flat_raw) == 0:
        return np.array([], dtype=target_dtype)

    if lut is not None:
        lut_size = len(lut)
        max_id = int(flat_raw.max()) if len(flat_raw) > 0 else 0
        if max_id >= lut_size:
            valid_flat = np.where(flat_raw < lut_size, flat_raw, 0)
            mapped_raw = lut[valid_flat]
        else:
            mapped_raw = lut[flat_raw]
    else:
        mapped_raw = flat_raw.astype(target_dtype, copy=False)

    if eos_id is None:
        return mapped_raw.astype(target_dtype, copy=False)

    num_docs = len(doc_lens)
    total_tokens = len(flat_raw) + num_docs
    result = np.full(total_tokens, new_eos_id, dtype=target_dtype)

    doc_idx = np.repeat(np.arange(num_docs, dtype=np.int64), doc_lens)
    write_indices = np.arange(len(flat_raw), dtype=np.int64) + doc_idx
    result[write_indices] = mapped_raw

    return result


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
    total_input_bytes = sum(os.path.getsize(p) for p in resolved_paths)
    total_input_mb = total_input_bytes / (1024 * 1024)
    for p in resolved_paths[:5]:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  - {p} ({size_mb:.2f} MB)")
    if len(resolved_paths) > 5:
        print(f"  ... and {len(resolved_paths) - 5} more files")
    print(f"Total input dataset size: {total_input_mb:.2f} MB")

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
            out_format = "bin"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    final_vocab_size = vocab_size
    orig_to_new = None
    new_to_orig = None

    # Step 3: Vocabulary Trimming Pass 1 (Streaming)
    if args.vocab_map_in:
        print(f"Applying existing vocabulary map from '{args.vocab_map_in}'...")
        vocab_meta, orig_to_new, new_to_orig = load_vocab_map(args.vocab_map_in)
        final_vocab_size = len(new_to_orig)
        print(f"  - Trimmed Vocab Size: {final_vocab_size:,} (Original: {vocab_size:,})")
    elif args.trim_vocab:
        print(f"Performing vocabulary trimming pass across input files (min_count={args.min_count})...")
        t_trim_start = time.time()
        counts = np.zeros(vocab_size, dtype=np.int64)

        for p in resolved_paths:
            for text_batch in stream_text_batches(p, file_type=file_type, json_field=args.json_field, parquet_column=args.parquet_column, separator=args.separator, batch_size=5000):
                if not text_batch:
                    continue
                batch_encoded = tokenizer.encode_batch(text_batch)
                if len(batch_encoded) == 0:
                    continue
                flat_file_raw = ak.to_numpy(ak.flatten(batch_encoded))
                if len(flat_file_raw) > 0:
                    max_id = int(flat_file_raw.max())
                    if max_id >= len(counts):
                        counts = np.pad(counts, (0, max_id + 1 - len(counts)))
                    counts += np.bincount(flat_file_raw, minlength=len(counts))
                del batch_encoded, flat_file_raw, text_batch
            gc.collect()

        new_to_orig, orig_to_new = build_trimmed_vocab_map_from_counts(
            counts,
            orig_vocab_size=vocab_size,
            always_keep_ids=[eos_id] if eos_id is not None else None,
            min_count=args.min_count,
        )
        final_vocab_size = len(new_to_orig)
        dead_tokens = vocab_size - final_vocab_size
        reduction_pct = (dead_tokens / vocab_size) * 100.0 if vocab_size > 0 else 0.0
        t_trim_end = time.time()

        print(f"Vocabulary Trimming Summary (in {t_trim_end - t_trim_start:.3f}s):")
        print(f"  - Original Vocab Size: {vocab_size:,}")
        print(f"  - Active (Used) Tokens: {final_vocab_size:,}")
        print(f"  - Min Count Threshold: {args.min_count}")
        print(f"  - Dead/Rare Tokens Removed: {dead_tokens:,} ({reduction_pct:.2f}%)")
        print(f"  - Embedding Matrix Size: ({final_vocab_size:,}, d_model)")

        map_out_path = args.vocab_map_out or os.path.join(os.path.dirname(os.path.abspath(args.output)), "vocab_map.json")
        save_vocab_map(map_out_path, orig_to_new, new_to_orig, orig_vocab_size=vocab_size, min_count=args.min_count)

    # Determine target dtype
    if args.dtype == "uint16" or (args.dtype == "auto" and final_vocab_size <= 65535):
        target_dtype = np.uint16
    else:
        target_dtype = np.uint32

    # Prepare Lookup Table (LUT) for remapping if active
    lut = None
    new_eos_id = eos_id if eos_id is not None else 0
    if orig_to_new is not None:
        max_orig_id = max(vocab_size, max(orig_to_new.keys()) + 1) if orig_to_new else vocab_size
        fallback_id = orig_to_new.get(eos_id, 0) if eos_id is not None else 0
        lut = np.full(max_orig_id, fallback_id, dtype=target_dtype)
        for orig_id, new_id in orig_to_new.items():
            if orig_id < max_orig_id:
                lut[orig_id] = new_id
        if eos_id is not None:
            new_eos_id = orig_to_new.get(eos_id, 0)

    # Step 4: Pass 2 - Streaming Tokenization & Export
    print(f"Exporting to '{args.output}' (format: {out_format}, dtype: {target_dtype.__name__})...")
    t_save_start = time.time()

    num_docs = 0
    raw_tok_count = 0
    final_tok_count = 0
    first_sample_doc = None

    if out_format == "bin":
        with open(args.output, "wb") as f_out:
            for p in resolved_paths:
                for text_batch in stream_text_batches(p, file_type=file_type, json_field=args.json_field, parquet_column=args.parquet_column, separator=args.separator, batch_size=5000):
                    if not text_batch:
                        continue
                    batch_encoded = tokenizer.encode_batch(text_batch)
                    if len(batch_encoded) == 0:
                        continue

                    if first_sample_doc is None:
                        first_sample_doc = np.array(batch_encoded[0])

                    doc_lens = ak.to_numpy(ak.num(batch_encoded))
                    flat_raw = ak.to_numpy(ak.flatten(batch_encoded))
                    raw_tok_count += len(flat_raw)
                    num_docs += len(doc_lens)

                    chunk_arr = flatten_and_remap_batch(
                        flat_raw=flat_raw,
                        doc_lens=doc_lens,
                        eos_id=eos_id if args.add_eos else None,
                        lut=lut,
                        target_dtype=target_dtype,
                        new_eos_id=new_eos_id,
                    )
                    if len(chunk_arr) > 0:
                        chunk_arr.tofile(f_out)
                        final_tok_count += len(chunk_arr)

                    del batch_encoded, flat_raw, doc_lens, chunk_arr, text_batch
                gc.collect()

    elif out_format == "npy":
        all_chunks = []
        for p in resolved_paths:
            for text_batch in stream_text_batches(p, file_type=file_type, json_field=args.json_field, parquet_column=args.parquet_column, separator=args.separator, batch_size=5000):
                if not text_batch:
                    continue
                batch_encoded = tokenizer.encode_batch(text_batch)
                if len(batch_encoded) == 0:
                    continue
                if first_sample_doc is None:
                    first_sample_doc = np.array(batch_encoded[0])
                doc_lens = ak.to_numpy(ak.num(batch_encoded))
                flat_raw = ak.to_numpy(ak.flatten(batch_encoded))
                raw_tok_count += len(flat_raw)
                num_docs += len(doc_lens)
                chunk_arr = flatten_and_remap_batch(
                    flat_raw=flat_raw,
                    doc_lens=doc_lens,
                    eos_id=eos_id if args.add_eos else None,
                    lut=lut,
                    target_dtype=target_dtype,
                    new_eos_id=new_eos_id,
                )
                if len(chunk_arr) > 0:
                    all_chunks.append(chunk_arr)
                del batch_encoded, flat_raw, doc_lens, text_batch
            gc.collect()

        if all_chunks:
            full_arr = np.concatenate(all_chunks)
            np.save(args.output, full_arr)
            final_tok_count = len(full_arr)

    elif out_format == "parquet":
        remapped_docs_list = []
        for p in resolved_paths:
            for text_batch in stream_text_batches(p, file_type=file_type, json_field=args.json_field, parquet_column=args.parquet_column, separator=args.separator, batch_size=5000):
                if not text_batch:
                    continue
                batch_encoded = tokenizer.encode_batch(text_batch)
                if len(batch_encoded) == 0:
                    continue
                if first_sample_doc is None:
                    first_sample_doc = np.array(batch_encoded[0])
                doc_lens = ak.to_numpy(ak.num(batch_encoded))
                flat_raw = ak.to_numpy(ak.flatten(batch_encoded))
                raw_tok_count += len(flat_raw)
                num_docs += len(doc_lens)
                if lut is not None:
                    for doc in batch_encoded:
                        doc_np = ak.to_numpy(doc)
                        valid_doc = np.where(doc_np < len(lut), doc_np, 0)
                        remapped_docs_list.append(lut[valid_doc])
                else:
                    for doc in batch_encoded:
                        remapped_docs_list.append(ak.to_numpy(doc).astype(target_dtype, copy=False))
                del batch_encoded, flat_raw, doc_lens, text_batch
            gc.collect()
        export_tokens = ak.Array(remapped_docs_list)
        ak.to_parquet(export_tokens, args.output)
        final_tok_count = int(ak.sum(ak.num(export_tokens)))

    t_save_end = time.time()
    out_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Export completed in {t_save_end - t_save_start:.3f}s!")
    print(f"  - Final output path: {args.output}")
    print(f"  - Output size: {out_size_mb:.2f} MB")
    print(f"  - Total token count: {final_tok_count:,}")
    print(f"  - Documents: {num_docs:,}")
    print(f"  - Raw tokens: {raw_tok_count:,}")

    # Decoded sanity check
    if first_sample_doc is not None and len(first_sample_doc) > 0:
        sample_doc = first_sample_doc
        sample_text = tokenizer.decode(sample_doc)
        print("\n--- Decoded Sample Check (Doc 1) ---")
        preview = sample_text[:200].decode("utf-8", errors="replace")
        print(f"Original Tokens (first 10): {sample_doc[:10]}")
        if orig_to_new is not None:
            trimmed_sample = [orig_to_new.get(int(tok), 0) for tok in sample_doc[:10]]
            print(f"Trimmed Tokens (first 10):  {trimmed_sample}")
            recovered = [int(new_to_orig[tok]) for tok in trimmed_sample if tok < len(new_to_orig)]
            decoded_back = tokenizer.decode(recovered).decode("utf-8", errors="replace")
            print(f"Decoded back from trimmed:  {decoded_back[:100]!r}...")
        else:
            print(f"Text preview: {preview!r}...")
        print("------------------------------------\n")


if __name__ == "__main__":
    main()

