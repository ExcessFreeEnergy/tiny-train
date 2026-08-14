#!/usr/bin/env python3
"""
prepare_dataset.py - Generalized Dataset Preparation Pipeline for Pretraining (FineWeb, FineWeb-Edu) & Finetuning (Open-Platypus).

Features:
  - FineWeb / FineWeb-Edu (Pretraining): Downloads HuggingFaceFW/fineweb or HuggingFaceFW/fineweb-edu parquet shards and tokenizes to 2.6B+ tokens.
  - Open-Platypus (Finetuning): Downloads garage-bAInd/Open-Platypus dataset, formats instruction-input-response
    prompts, and tokenizes into train (~95%) and validation (~5%) datasets.
  - Vocabulary Alignment: For Open-Platypus, automatically inspects pretraining vocab_map.json if present
    to align token IDs with the pre-trained embedding vocabulary.
  - Idempotency & Verification: Automatically verifies existing binary dataset artifacts (.bin and vocab_map.json).
  - Automated Remediation: Detects corrupted or missing dataset artifacts, cleans stale files, and remediates.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from huggingface_hub import HfApi, hf_hub_download


def verify_dataset(data_dir: str, min_train_tokens: int = 10_000, min_valid_tokens: int = 1_000) -> bool:
    """Verify that dataset binaries and vocab_map.json exist, are non-empty, and contain sufficient valid tokens."""
    data_dir = os.path.abspath(data_dir)
    train_bin = os.path.join(data_dir, "train_trimmed.bin")
    if not os.path.exists(train_bin):
        train_bin = os.path.join(data_dir, "train.bin")

    valid_bin = os.path.join(data_dir, "valid_trimmed.bin")
    if not os.path.exists(valid_bin):
        valid_bin = os.path.join(data_dir, "valid.bin")

    vocab_map = os.path.join(data_dir, "vocab_map.json")

    # Check existence
    if not os.path.exists(train_bin):
        print(f"⚠️ Verification check failed: missing training binary file in {data_dir}", flush=True)
        return False
    if not os.path.exists(valid_bin):
        print(f"⚠️ Verification check failed: missing validation binary file in {data_dir}", flush=True)
        return False
    if not os.path.exists(vocab_map):
        print(f"⚠️ Verification check failed: missing vocab_map.json in {data_dir}", flush=True)
        return False

    # Check file sizes and uint16 alignment (2 bytes per token)
    train_sz = os.path.getsize(train_bin)
    valid_sz = os.path.getsize(valid_bin)
    if train_sz == 0 or train_sz % 2 != 0:
        print(f"⚠️ Verification check failed: train binary '{train_bin}' is empty or invalid size ({train_sz} bytes)", flush=True)
        return False
    if valid_sz == 0 or valid_sz % 2 != 0:
        print(f"⚠️ Verification check failed: valid binary '{valid_bin}' is empty or invalid size ({valid_sz} bytes)", flush=True)
        return False

    # Check memmap readability and token counts
    try:
        train_arr = np.memmap(train_bin, dtype=np.uint16, mode="r")
        train_tokens = len(train_arr)
        if train_tokens < min_train_tokens:
            print(f"⚠️ Verification check failed: '{train_bin}' token count ({train_tokens:,}) < min required ({min_train_tokens:,})", flush=True)
            return False
    except Exception as e:
        print(f"⚠️ Verification check failed: unable to memmap train binary '{train_bin}': {e}", flush=True)
        return False

    try:
        valid_arr = np.memmap(valid_bin, dtype=np.uint16, mode="r")
        valid_tokens = len(valid_arr)
        if valid_tokens < min_valid_tokens:
            print(f"⚠️ Verification check failed: '{valid_bin}' token count ({valid_tokens:,}) < min required ({min_valid_tokens:,})", flush=True)
            return False
    except Exception as e:
        print(f"⚠️ Verification check failed: unable to memmap valid binary '{valid_bin}': {e}", flush=True)
        return False

    # Check vocab_map.json JSON validity
    try:
        with open(vocab_map) as f:
            vdata = json.load(f)
            if "trimmed_vocab_size" not in vdata and "original_vocab_size" not in vdata and "active_vocab_size" not in vdata:
                print(f"⚠️ Verification check failed: '{vocab_map}' missing vocabulary metadata keys", flush=True)
                return False
    except Exception as e:
        print(f"⚠️ Verification check failed: invalid JSON in '{vocab_map}': {e}", flush=True)
        return False

    return True


def remediate_dataset(data_dir: str):
    """Clean up corrupted or stale generated dataset artifacts in data_dir while preserving raw parquet downloads."""
    data_dir = os.path.abspath(data_dir)
    print(f"🛠️ Remediating dataset directory '{data_dir}'...", flush=True)

    stale_files = [
        "train_trimmed.bin",
        "valid_trimmed.bin",
        "train.bin",
        "valid.bin",
        "full_trimmed.bin",
        "vocab_map.json",
        "formatted.jsonl",
    ]

    for fname in stale_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.islink(fpath) or os.path.exists(fpath):
            try:
                os.remove(fpath)
                print(f"  - Removed stale file: {fpath}", flush=True)
            except Exception as e:
                print(f"  - Warning: Failed to remove {fpath}: {e}", flush=True)


def download_fineweb_shards(raw_dir: str, target_tokens: int = 2_605_000_000, repo_id: str = "HuggingFaceFW/fineweb") -> list[str]:
    """Download FineWeb or FineWeb-Edu parquet shards until target token count is reachable."""
    os.makedirs(raw_dir, exist_ok=True)
    # Estimate ~275M tokens per shard for Cosmopedia-v2, ~650M for FineWeb
    tokens_per_shard = 275_000_000 if "cosmopedia" in repo_id.lower() else 650_000_000
    needed_shards = max(2, int(np.ceil(target_tokens / tokens_per_shard)))

    existing_parquets = glob.glob(os.path.join(raw_dir, "**/*.parquet"), recursive=True)
    valid_parquets = [p for p in existing_parquets if os.path.exists(p) and os.path.getsize(p) > 0]
    if len(valid_parquets) >= needed_shards:
        print(f"Found {len(valid_parquets)} existing parquet shard(s) in {raw_dir} (needed {needed_shards}):", flush=True)
        for p in valid_parquets:
            sz_mb = os.path.getsize(p) / (1024**2)
            print(f"  - {p} ({sz_mb:.1f} MB)", flush=True)
        return valid_parquets

    print(f"🔍 Fetching shard list from {repo_id} for {target_tokens:,} target tokens ({needed_shards} shards needed)...", flush=True)
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Prefer sample/10BT/, sample/100BT/, sample/350BT/, cosmopedia-v2/, or data/ parquet files
    sample_10bt = sorted([f for f in files if f.startswith("sample/10BT/") and f.endswith(".parquet")])
    shards = sample_10bt if sample_10bt else sorted([f for f in files if f.endswith(".parquet") and not f.startswith(".")])

    print(f"Found {len(shards)} parquet shards in {repo_id}.", flush=True)

    downloaded_files = list(valid_parquets)
    for shard_path in shards:
        actual_path = os.path.join(raw_dir, shard_path)
        filename = os.path.basename(shard_path)

        if actual_path in downloaded_files:
            continue

        if os.path.exists(actual_path) and os.path.getsize(actual_path) > 0:
            print(f"  - Shard already exists: {actual_path} ({os.path.getsize(actual_path) / (1024**2):.1f} MB)", flush=True)
            downloaded_files.append(actual_path)
        else:
            print(f"  - Downloading shard [{len(downloaded_files) + 1}/{needed_shards}]: {filename}...", flush=True)
            t0 = time.time()
            downloaded_file = None
            for attempt in range(5):
                try:
                    downloaded_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=shard_path,
                        repo_type="dataset",
                        local_dir=raw_dir,
                    )
                    break
                except Exception as e:
                    print(f"    ⚠️ Download attempt {attempt + 1}/5 failed ({e}). Retrying in 3s...", flush=True)
                    time.sleep(3)
            if not downloaded_file or not os.path.exists(downloaded_file):
                raise RuntimeError(f"Failed to download shard {shard_path} after 5 attempts.")

            dt = time.time() - t0
            sz_mb = os.path.getsize(downloaded_file) / (1024**2)
            print(f"    Downloaded in {dt:.2f}s ({sz_mb:.1f} MB)", flush=True)
            downloaded_files.append(downloaded_file)

        if len(downloaded_files) >= needed_shards:
            print(f"Downloaded {len(downloaded_files)} shards (~{len(downloaded_files) * 650}M tokens estimated).", flush=True)
            break

    return downloaded_files


def download_open_platypus_shards(raw_dir: str) -> list[str]:
    """Download garage-bAInd/Open-Platypus parquet dataset files."""
    os.makedirs(raw_dir, exist_ok=True)

    existing_parquets = glob.glob(os.path.join(raw_dir, "**/*.parquet"), recursive=True)
    valid_parquets = [p for p in existing_parquets if os.path.exists(p) and os.path.getsize(p) > 0]
    if valid_parquets:
        print(f"Found {len(valid_parquets)} existing parquet shard(s) for Open-Platypus in {raw_dir}:", flush=True)
        for p in valid_parquets:
            sz_mb = os.path.getsize(p) / (1024**2)
            print(f"  - {p} ({sz_mb:.1f} MB)", flush=True)
        return valid_parquets

    print("🔍 Fetching dataset file list from garage-bAInd/Open-Platypus...", flush=True)
    api = HfApi()
    files = api.list_repo_files(repo_id="garage-bAInd/Open-Platypus", repo_type="dataset")
    shards = sorted([f for f in files if f.endswith(".parquet")])

    if not shards:
        raise RuntimeError("No parquet shards found in repository garage-bAInd/Open-Platypus.")

    downloaded_files = []
    for shard_path in shards:
        filename = os.path.basename(shard_path)
        print(f"  - Downloading Open-Platypus shard: {filename}...", flush=True)
        t0 = time.time()
        downloaded_file = hf_hub_download(
            repo_id="garage-bAInd/Open-Platypus",
            filename=shard_path,
            repo_type="dataset",
            local_dir=raw_dir,
        )
        dt = time.time() - t0
        sz_mb = os.path.getsize(downloaded_file) / (1024**2)
        print(f"    Downloaded in {dt:.2f}s ({sz_mb:.1f} MB)", flush=True)
        downloaded_files.append(downloaded_file)

    return downloaded_files


def format_open_platypus_prompt(instruction: str, input_text: str, output_text: str) -> str:
    """Format an Open-Platypus row into a standard instruction prompt."""
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    output_text = (output_text or "").strip()

    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )


def format_open_platypus_jsonl(parquet_paths: list[str], output_jsonl: str) -> int:
    """Format raw Open-Platypus parquet rows into JSONL text documents."""
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    print(f"✍️ Formatting Open-Platypus records to '{output_jsonl}'...", flush=True)

    total_records = 0
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for ppath in parquet_paths:
            table = pq.read_table(ppath)
            cols = table.column_names
            num_rows = table.num_rows

            instructions = table["instruction"].to_pylist() if "instruction" in cols else [""] * num_rows
            inputs = table["input"].to_pylist() if "input" in cols else [""] * num_rows
            outputs = table["output"].to_pylist() if "output" in cols else [""] * num_rows

            for inst, inp, out in zip(instructions, inputs, outputs):
                formatted_text = format_open_platypus_prompt(inst, inp, out)
                record = json.dumps({"text": formatted_text})
                f_out.write(record + "\n")
                total_records += 1

    print(f"Formatted {total_records:,} Open-Platypus prompt entries into JSONL.", flush=True)
    return total_records


def prepare_pretraining_dataset(
    target_dir: str,
    repo_id: str = "HuggingFaceFW/fineweb",
    dataset_name: str = "FineWeb",
    target_tokens: int = 2_600_000_000,
    valid_tokens: int = 5_000_000,
    min_count: int = 50,
    force: bool = False,
):
    """Download, tokenize, split, verify, and remediate pretraining datasets (FineWeb, FineWeb-Edu)."""
    target_dir = os.path.abspath(target_dir)
    raw_dir = os.path.join(target_dir, "raw")

    print(f"\n=== {dataset_name} Pretraining Dataset Pipeline ===")
    print(f"Repository: {repo_id}")
    print(f"Target Directory: {target_dir}")
    print(f"Target Train Tokens: {target_tokens:,}")
    print(f"Target Valid Tokens: {valid_tokens:,}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=target_tokens, min_valid_tokens=valid_tokens):
        print(f"[OK] Dataset '{dataset_name}' in '{target_dir}' verified and ready to consume! Skipping regeneration.", flush=True)
        return

    if force:
        print(f"Force flag specified. Re-generating {dataset_name} dataset...", flush=True)
    else:
        print(f"{dataset_name} dataset missing or corrupted. Triggering remediation...", flush=True)

    remediate_dataset(target_dir)

    # Step 1: Download parquet shards
    downloaded_shards = download_fineweb_shards(raw_dir, target_tokens=target_tokens + valid_tokens, repo_id=repo_id)
    if not downloaded_shards:
        raise RuntimeError(f"No {dataset_name} parquet shards downloaded.")

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
        str(min_count),
        "--vocab-map-out",
        vocab_map,
    ]

    print(f"\n🚀 Tokenizing {dataset_name} shards using Gigatoken engine...", flush=True)
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

    needed_total = target_tokens + valid_tokens
    if total_tokens < needed_total:
        raise ValueError(f"Generated tokens ({total_tokens:,}) less than required ({needed_total:,}). Download more shards.")

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")

    print(f"\nSplitting into train_trimmed.bin ({target_tokens:,} tokens) and valid_trimmed.bin ({valid_tokens:,} tokens)...", flush=True)

    train_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(target_tokens,))
    chunk_size = 50_000_000
    for offset in range(0, target_tokens, chunk_size):
        end = min(offset + chunk_size, target_tokens)
        train_mmap[offset:end] = full_data[offset:end]
    train_mmap.flush()
    del train_mmap

    valid_mmap = np.memmap(valid_bin, dtype=np.uint16, mode="w+", shape=(valid_tokens,))
    for offset in range(0, valid_tokens, chunk_size):
        end = min(offset + chunk_size, valid_tokens)
        valid_mmap[offset:end] = full_data[target_tokens + offset : target_tokens + end]
    valid_mmap.flush()
    del valid_mmap

    # Write fallbacks / symlinks
    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)

    # Cleanup temporary full_bin
    if os.path.exists(full_bin):
        os.remove(full_bin)

    # Verification
    if not verify_dataset(target_dir, min_train_tokens=target_tokens, min_valid_tokens=valid_tokens):
        raise RuntimeError(f"Post-generation verification failed for {dataset_name} in '{target_dir}'.")

    print(f"\n✅ {dataset_name} Pretraining Dataset Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {target_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB, {valid_tokens:,} tokens)")
    print(f"  - Vocab Map:     '{vocab_map}'")


def prepare_open_platypus(
    target_dir: str,
    min_count: int = 50,
    force: bool = False,
    fineweb_dir: str = "data/FineWeb",
):
    """Download, format, tokenize, split, verify, and remediate Open-Platypus finetuning dataset."""
    target_dir = os.path.abspath(target_dir)
    raw_dir = os.path.join(target_dir, "raw")

    print("\n=== Open-Platypus Finetuning Dataset Pipeline ===")
    print(f"Target Directory: {target_dir}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=100_000, min_valid_tokens=10_000):
        print(f"[OK] Dataset 'OpenPlatypus' in '{target_dir}' verified and ready to consume! Skipping regeneration.", flush=True)
        return

    if force:
        print("Force flag specified. Re-generating Open-Platypus dataset...", flush=True)
    else:
        print("Open-Platypus dataset missing or corrupted. Triggering remediation...", flush=True)

    remediate_dataset(target_dir)

    # Step 1: Download Open-Platypus shards
    shards = download_open_platypus_shards(raw_dir)
    if not shards:
        raise RuntimeError("No Open-Platypus parquet shards downloaded.")

    # Step 2: Format records into JSONL text documents
    formatted_jsonl = os.path.join(raw_dir, "formatted.jsonl")
    total_records = format_open_platypus_jsonl(shards, formatted_jsonl)
    if total_records == 0:
        raise RuntimeError("Formatted zero records from Open-Platypus dataset.")

    # Step 3: Tokenize using retokenize.py
    full_bin = os.path.join(target_dir, "full_trimmed.bin")
    vocab_map = os.path.join(target_dir, "vocab_map.json")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retokenize_script = os.path.join(project_root, "src", "retokenize.py")

    fineweb_vocab_map = os.path.abspath(os.path.join(fineweb_dir, "vocab_map.json"))

    cmd = [
        sys.executable,
        retokenize_script,
        "-i",
        formatted_jsonl,
        "-o",
        full_bin,
        "--file-type",
        "jsonl",
        "--json-field",
        "text",
    ]

    if os.path.exists(fineweb_vocab_map):
        print(f"🔗 Aligning Open-Platypus vocabulary with pretraining vocab map '{fineweb_vocab_map}'...", flush=True)
        cmd.extend(["--vocab-map-in", fineweb_vocab_map])
        shutil.copy(fineweb_vocab_map, vocab_map)
    else:
        print("⚠️ Pretraining vocab_map.json not found. Generating standalone trimmed vocabulary for Open-Platypus...", flush=True)
        cmd.extend(["--trim-vocab", "--min-count", str(min_count), "--vocab-map-out", vocab_map])

    print("\n🚀 Tokenizing Open-Platypus formatted prompts using Gigatoken engine...", flush=True)
    t_tok0 = time.time()
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Retokenization failed for Open-Platypus with exit code {res.returncode}")
    print(f"Tokenization complete in {time.time() - t_tok0:.2f}s!", flush=True)

    if not os.path.exists(full_bin):
        raise FileNotFoundError(f"Expected binary output '{full_bin}' not found.")

    full_data = np.memmap(full_bin, dtype=np.uint16, mode="r")
    total_tokens = len(full_data)
    print(f"\nTotal tokens generated: {total_tokens:,} ({os.path.getsize(full_bin) / (1024**2):.2f} MB)", flush=True)

    # Step 4: Partition into 95% train / 5% valid split
    train_tokens = int(total_tokens * 0.95)
    valid_tokens = total_tokens - train_tokens

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")

    print(f"\nSplitting into train_trimmed.bin ({train_tokens:,} tokens) and valid_trimmed.bin ({valid_tokens:,} tokens)...", flush=True)

    train_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(train_tokens,))
    chunk_size = 50_000_000
    for offset in range(0, train_tokens, chunk_size):
        end = min(offset + chunk_size, train_tokens)
        train_mmap[offset:end] = full_data[offset:end]
    train_mmap.flush()
    del train_mmap

    valid_mmap = np.memmap(valid_bin, dtype=np.uint16, mode="w+", shape=(valid_tokens,))
    for offset in range(0, valid_tokens, chunk_size):
        end = min(offset + chunk_size, valid_tokens)
        valid_mmap[offset:end] = full_data[train_tokens + offset : train_tokens + end]
    valid_mmap.flush()
    del valid_mmap

    # Write fallbacks / symlinks
    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)

    # Cleanup temporary full_bin and formatted.jsonl
    if os.path.exists(full_bin):
        os.remove(full_bin)
    if os.path.exists(formatted_jsonl):
        os.remove(formatted_jsonl)

    # Verification
    if not verify_dataset(target_dir, min_train_tokens=train_tokens, min_valid_tokens=valid_tokens):
        raise RuntimeError(f"Post-generation verification failed for Open-Platypus in '{target_dir}'.")

    print("\n✅ Open-Platypus Finetuning Dataset Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {train_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB, {valid_tokens:,} tokens)")
    print(f"  - Vocab Map:     '{vocab_map}'")


def main():
    parser = argparse.ArgumentParser(description="Generalized Dataset Preparation Pipeline for Pretraining & Finetuning.")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["all", "fineweb", "fineweb-edu", "cosmopedia", "cosmopedia-v2", "platypus", "open-platypus"],
        default="all",
        help="Dataset(s) to download and prepare (default: all)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Target root directory override for dataset output")
    parser.add_argument("--fineweb-dir", type=str, default="data/FineWeb", help="Directory for FineWeb pretraining dataset")
    parser.add_argument("--fineweb-edu-dir", type=str, default="data/FineWebEdu", help="Directory for FineWeb-Edu pretraining dataset")
    parser.add_argument("--cosmopedia-dir", type=str, default="data/CosmopediaV2", help="Directory for Cosmopedia v2 pretraining dataset")
    parser.add_argument("--platypus-dir", type=str, default="data/OpenPlatypus", help="Directory for Open-Platypus dataset")
    parser.add_argument("--target-tokens", type=int, default=2_600_000_000, help="Target pretraining train token count (default: 2.6B)")
    parser.add_argument("--valid-tokens", type=int, default=5_000_000, help="Target pretraining valid token count (default: 5M)")
    parser.add_argument("--min-count", type=int, default=50, help="Vocabulary trimming minimum count threshold")
    parser.add_argument("--force", action="store_true", default=False, help="Force dataset re-download and re-tokenization")

    args = parser.parse_args()

    dataset_choice = args.dataset.lower()

    fineweb_path = args.data_dir if (args.data_dir and dataset_choice == "fineweb") else args.fineweb_dir
    fineweb_edu_path = args.data_dir if (args.data_dir and dataset_choice == "fineweb-edu") else args.fineweb_edu_dir
    cosmopedia_path = args.data_dir if (args.data_dir and dataset_choice in ["cosmopedia", "cosmopedia-v2"]) else args.cosmopedia_dir
    platypus_path = args.data_dir if (args.data_dir and dataset_choice in ["platypus", "open-platypus"]) else args.platypus_dir

    if dataset_choice in ["all", "fineweb"]:
        prepare_pretraining_dataset(
            target_dir=fineweb_path,
            repo_id="HuggingFaceFW/fineweb",
            dataset_name="FineWeb",
            target_tokens=args.target_tokens,
            valid_tokens=args.valid_tokens,
            min_count=args.min_count,
            force=args.force,
        )

    if dataset_choice in ["all", "fineweb-edu"]:
        prepare_pretraining_dataset(
            target_dir=fineweb_edu_path,
            repo_id="HuggingFaceFW/fineweb-edu",
            dataset_name="FineWeb-Edu",
            target_tokens=args.target_tokens,
            valid_tokens=args.valid_tokens,
            min_count=args.min_count,
            force=args.force,
        )

    if dataset_choice in ["all", "cosmopedia", "cosmopedia-v2"]:
        prepare_pretraining_dataset(
            target_dir=cosmopedia_path,
            repo_id="HuggingFaceTB/cosmopedia-v2",
            dataset_name="Cosmopedia-v2",
            target_tokens=args.target_tokens,
            valid_tokens=args.valid_tokens,
            min_count=args.min_count,
            force=args.force,
        )

    if dataset_choice in ["all", "platypus", "open-platypus"]:
        prepare_open_platypus(
            target_dir=platypus_path,
            min_count=args.min_count,
            force=args.force,
            fineweb_dir=fineweb_edu_path if os.path.exists(fineweb_edu_path) else fineweb_path,
        )


if __name__ == "__main__":
    main()
