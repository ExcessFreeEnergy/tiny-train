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
    # Estimate ~120M tokens per shard for BookCorpus, ~275M for Cosmopedia-v2, ~650M for FineWeb
    if "bookcorpus" in repo_id.lower():
        tokens_per_shard = 120_000_000
    elif "cosmopedia" in repo_id.lower():
        tokens_per_shard = 275_000_000
    else:
        tokens_per_shard = 650_000_000
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
    vocab_map_in: str | None = None,
):
    """Download parquet shards and tokenize into binary memmaps."""
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
    ]

    if vocab_map_in and os.path.exists(vocab_map_in):
        print(f"🔗 Aligning {dataset_name} vocabulary with existing vocab map '{vocab_map_in}'...", flush=True)
        cmd.extend(["--vocab-map-in", vocab_map_in])
        shutil.copy(vocab_map_in, vocab_map)
    else:
        cmd.extend(["--trim-vocab", "--min-count", str(min_count), "--vocab-map-out", vocab_map])

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
        actual_train_tokens = max(100_000, total_tokens - valid_tokens)
        print(
            f"⚠️ Note: Total dataset tokens ({total_tokens:,}) is smaller than target ({needed_total:,}). "
            f"Using maximum available tokens for training split: {actual_train_tokens:,} tokens.",
            flush=True,
        )
        target_tokens = actual_train_tokens

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


def prepare_custom_blend(
    target_dir: str = "data/BookCorpusFineWebEdu",
    bookcorpus_dir: str = "data/BookCorpus",
    fineweb_edu_dir: str = "data/FineWebEdu",
    target_train_tokens: int = 2_600_000_000,
    target_valid_tokens: int = 5_000_000,
    bookcorpus_ratio: float = 0.70,
    force: bool = False,
):
    """Combine ~70% BookCorpus and ~30% FineWeb-Edu tokens to form a 2.6B token pretraining dataset."""
    target_dir = os.path.abspath(target_dir)
    print("\n=== Custom Blend Dataset Pipeline (~70% BookCorpus / ~30% FineWeb-Edu) ===")
    print(f"Target Directory: {target_dir}")
    print(f"Target Train Tokens: {target_train_tokens:,}")
    print(f"Target Valid Tokens: {target_valid_tokens:,}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=target_train_tokens, min_valid_tokens=target_valid_tokens):
        print(f"[OK] Dataset 'BookCorpusFineWebEdu' in '{target_dir}' verified and ready to consume! Skipping generation.", flush=True)
        return

    # Ensure source datasets are prepared
    if not verify_dataset(bookcorpus_dir, min_train_tokens=1_000_000, min_valid_tokens=1_000):
        print("⚡ BookCorpus source dataset missing or incomplete. Triggering preparation...", flush=True)
        prepare_pretraining_dataset(
            target_dir=bookcorpus_dir,
            repo_id="lucadiliello/bookcorpusopen",
            dataset_name="BookCorpus",
            target_tokens=2_600_000_000,
            valid_tokens=5_000_000,
            force=force,
        )

    bc_vocab_map = os.path.join(bookcorpus_dir, "vocab_map.json")
    if not verify_dataset(fineweb_edu_dir, min_train_tokens=1_000_000, min_valid_tokens=1_000):
        print("⚡ FineWeb-Edu source dataset missing or incomplete. Triggering aligned preparation...", flush=True)
        prepare_pretraining_dataset(
            target_dir=fineweb_edu_dir,
            repo_id="HuggingFaceFW/fineweb-edu",
            dataset_name="FineWeb-Edu",
            target_tokens=2_600_000_000,
            valid_tokens=5_000_000,
            force=force,
            vocab_map_in=bc_vocab_map if os.path.exists(bc_vocab_map) else None,
        )

    os.makedirs(target_dir, exist_ok=True)
    remediate_dataset(target_dir)

    bc_train_path = os.path.join(bookcorpus_dir, "train_trimmed.bin")
    if not os.path.exists(bc_train_path):
        bc_train_path = os.path.join(bookcorpus_dir, "train.bin")

    fe_train_path = os.path.join(fineweb_edu_dir, "train_trimmed.bin")
    if not os.path.exists(fe_train_path):
        fe_train_path = os.path.join(fineweb_edu_dir, "train.bin")

    fe_vocab_map = os.path.join(fineweb_edu_dir, "vocab_map.json")
    target_vocab_map = os.path.join(target_dir, "vocab_map.json")
    if os.path.exists(fe_vocab_map):
        shutil.copy(fe_vocab_map, target_vocab_map)

    bc_mmap = np.memmap(bc_train_path, dtype=np.uint16, mode="r")
    fe_mmap = np.memmap(fe_train_path, dtype=np.uint16, mode="r")

    bc_req = int(target_train_tokens * bookcorpus_ratio)
    fe_req = target_train_tokens - bc_req

    bc_count = min(len(bc_mmap), bc_req)
    fe_count = min(len(fe_mmap), fe_req)

    actual_train_tokens = bc_count + fe_count
    print(f"Blending {bc_count:,} BookCorpus tokens + {fe_count:,} FineWeb-Edu tokens = {actual_train_tokens:,} total train tokens...", flush=True)
    print("🔀 Performing sequence-block shuffling (2,048 tokens per chunk) to interleave domains...", flush=True)

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")

    # Group into 2048-token sequence blocks to preserve local text coherency while globally interleaving sources
    block_size = 2048
    bc_num_blocks = bc_count // block_size
    fe_num_blocks = fe_count // block_size

    # Build index array of (source_dataset, block_index) tuples
    blocks = [("bc", i) for i in range(bc_num_blocks)] + [("fe", i) for i in range(fe_num_blocks)]
    rng = np.random.default_rng(seed=42)
    rng.shuffle(blocks)

    total_shuffled_blocks = len(blocks)
    actual_train_tokens = total_shuffled_blocks * block_size

    out_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(actual_train_tokens,))

    chunk_blocks = 50_000  # Process in ~100M token batches
    for i in range(0, total_shuffled_blocks, chunk_blocks):
        batch = blocks[i : i + chunk_blocks]
        out_offset = i * block_size

        for b_idx, (src, src_blk_idx) in enumerate(batch):
            src_mmap = bc_mmap if src == "bc" else fe_mmap
            src_start = src_blk_idx * block_size
            src_end = src_start + block_size

            dst_start = out_offset + (b_idx * block_size)
            dst_end = dst_start + block_size

            out_mmap[dst_start:dst_end] = src_mmap[src_start:src_end]

    out_mmap.flush()
    del out_mmap

    # Copy validation split from FineWeb-Edu
    fe_valid_path = os.path.join(fineweb_edu_dir, "valid_trimmed.bin")
    if not os.path.exists(fe_valid_path):
        fe_valid_path = os.path.join(fineweb_edu_dir, "valid.bin")

    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")
    shutil.copy(fe_valid_path, valid_bin)

    # Symlinks/Fallbacks
    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)

    if not verify_dataset(target_dir, min_train_tokens=actual_train_tokens, min_valid_tokens=1_000):
        raise RuntimeError(f"Post-generation verification failed for BookCorpusFineWebEdu in '{target_dir}'.")

    print("\n✅ Custom Blend (BookCorpus + FineWeb-Edu) Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {actual_train_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB)")
    print(f"  - Vocab Map:     '{target_vocab_map}'")


def download_synth_apigen_shards(raw_dir: str) -> list[str]:
    """Download argilla/Synth-APIGen-v0.1 dataset parquet file."""
    os.makedirs(raw_dir, exist_ok=True)
    existing = glob.glob(os.path.join(raw_dir, "**/*.parquet"), recursive=True)
    valid = [p for p in existing if os.path.exists(p) and os.path.getsize(p) > 0]
    if valid:
        print(f"Found {len(valid)} existing parquet shard(s) for Synth-APIGen in {raw_dir}:", flush=True)
        for p in valid:
            print(f"  - {p} ({os.path.getsize(p) / (1024**2):.1f} MB)", flush=True)
        return valid

    print("🔍 Downloading argilla/Synth-APIGen-v0.1 dataset...", flush=True)
    fpath = hf_hub_download(
        repo_id="argilla/Synth-APIGen-v0.1",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=raw_dir,
    )
    return [fpath]


def format_synth_apigen_jsonl(parquet_paths: list[str], output_jsonl: str) -> int:
    """Format argilla/Synth-APIGen-v0.1 parquet rows into tool-calling instruction JSONL records, preserving native format."""
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    print(f"✍️ Formatting Synth-APIGen records to '{output_jsonl}'...", flush=True)

    total_records = 0
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for ppath in parquet_paths:
            table = pq.read_table(ppath)
            cols = table.column_names
            num_rows = table.num_rows

            queries = table["query"].to_pylist() if "query" in cols else [""] * num_rows
            answers = table["answers"].to_pylist() if "answers" in cols else [""] * num_rows
            tools_col = table["tools"].to_pylist() if "tools" in cols else [""] * num_rows

            for query_text, ans_text, tools_text in zip(queries, answers, tools_col):
                query_str = (query_text or "").strip()
                ans_str = (ans_text or "").strip()
                tools_str = (tools_text or "").strip()

                # Filter out unanswerable queries
                if not ans_str or ans_str.startswith("The query cannot be answered") or ans_str.startswith("The given question lacks"):
                    continue

                if tools_str and tools_str != "[]":
                    prompt_text = (
                        "Below is a user query along with available tool definitions in JSON format. "
                        "Select the appropriate tool(s) and provide the tool call(s) as a strict JSON array.\n\n"
                        f"<tools>\n{tools_str}\n</tools>\n\n"
                        f"### Query:\n{query_str}\n\n"
                        f"### Response:\n{ans_str}"
                    )
                else:
                    prompt_text = (
                        "Below is a user query. Provide the appropriate tool call or structured JSON response.\n\n"
                        f"### Query:\n{query_str}\n\n"
                        f"### Response:\n{ans_str}"
                    )

                record = json.dumps({"text": prompt_text})
                f_out.write(record + "\n")
                total_records += 1

    print(f"Formatted {total_records:,} Synth-APIGen instruction entries into JSONL.", flush=True)
    return total_records


def prepare_synth_apigen(
    target_dir: str = "data/SynthAPIGen",
    min_count: int = 50,
    force: bool = False,
    vocab_map_in: str | None = None,
):
    """Download, format, tokenize, split, verify, and remediate Synth-APIGen tool-calling dataset."""
    target_dir = os.path.abspath(target_dir)
    raw_dir = os.path.join(target_dir, "raw")

    print("\n=== Argilla Synth-APIGen Tool-Calling Dataset Pipeline ===")
    print(f"Target Directory: {target_dir}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=50_000, min_valid_tokens=5_000):
        print(f"[OK] Dataset 'SynthAPIGen' in '{target_dir}' verified and ready to consume! Skipping regeneration.", flush=True)
        return

    if force:
        print("Force flag specified. Re-generating Synth-APIGen dataset...", flush=True)
    else:
        print("Synth-APIGen dataset missing or corrupted. Triggering remediation...", flush=True)

    remediate_dataset(target_dir)

    # Step 1: Download parquet shards
    shards = download_synth_apigen_shards(raw_dir)

    # Step 2: Format records into JSONL
    formatted_jsonl = os.path.join(raw_dir, "formatted.jsonl")
    total_records = format_synth_apigen_jsonl(shards, formatted_jsonl)
    if total_records == 0:
        raise RuntimeError("Formatted zero records from Synth-APIGen dataset.")

    # Step 3: Tokenize using retokenize.py
    full_bin = os.path.join(target_dir, "full_trimmed.bin")
    vocab_map = os.path.join(target_dir, "vocab_map.json")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retokenize_script = os.path.join(project_root, "src", "retokenize.py")

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

    if vocab_map_in and os.path.exists(vocab_map_in):
        print(f"🔗 Aligning Synth-APIGen vocabulary with pretraining vocab map '{vocab_map_in}'...", flush=True)
        cmd.extend(["--vocab-map-in", vocab_map_in])
        shutil.copy(vocab_map_in, vocab_map)
    else:
        print("Generating vocabulary map for Synth-APIGen...", flush=True)
        cmd.extend(["--trim-vocab", "--min-count", str(min_count), "--vocab-map-out", vocab_map])

    print("\n🚀 Tokenizing Synth-APIGen formatted prompts using Gigatoken engine...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Retokenization failed for Synth-APIGen with exit code {res.returncode}")
    print(f"Tokenization complete in {time.time() - t0:.2f}s!", flush=True)

    full_data = np.memmap(full_bin, dtype=np.uint16, mode="r")
    total_tokens = len(full_data)

    # Step 4: Split into 95% train / 5% valid split
    train_tokens = int(total_tokens * 0.95)
    valid_tokens = total_tokens - train_tokens

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")

    train_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(train_tokens,))
    train_mmap[:] = full_data[:train_tokens]
    train_mmap.flush()
    del train_mmap

    valid_mmap = np.memmap(valid_bin, dtype=np.uint16, mode="w+", shape=(valid_tokens,))
    valid_mmap[:] = full_data[train_tokens : train_tokens + valid_tokens]
    valid_mmap.flush()
    del valid_mmap

    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)

    if os.path.exists(full_bin):
        os.remove(full_bin)
    if os.path.exists(formatted_jsonl):
        os.remove(formatted_jsonl)

    if not verify_dataset(target_dir, min_train_tokens=train_tokens, min_valid_tokens=valid_tokens):
        raise RuntimeError(f"Post-generation verification failed for Synth-APIGen in '{target_dir}'.")

    print("\n✅ Synth-APIGen Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {train_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB, {valid_tokens:,} tokens)")


def download_hermes_function_calling_shards(raw_dir: str) -> list[str]:
    """Download NousResearch/hermes-function-calling-v1 JSON files."""
    os.makedirs(raw_dir, exist_ok=True)
    files = [
        "func-calling.json",
        "glaive-function-calling-5k.json",
        "json-mode-agentic.json",
        "json-mode-singleturn.json",
        "func-calling-singleturn.json",
    ]

    downloaded = []
    print("🔍 Downloading NousResearch/hermes-function-calling-v1 files...", flush=True)
    for fname in files:
        target_path = os.path.join(raw_dir, fname)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"  - Already exists: {target_path} ({os.path.getsize(target_path) / (1024**2):.1f} MB)", flush=True)
            downloaded.append(target_path)
        else:
            try:
                path = hf_hub_download(
                    repo_id="NousResearch/hermes-function-calling-v1",
                    filename=fname,
                    repo_type="dataset",
                    local_dir=raw_dir,
                )
                downloaded.append(path)
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to download {fname}: {e}", flush=True)

    return downloaded


def format_hermes_jsonl(json_paths: list[str], output_jsonl: str) -> int:
    """Format NousResearch/hermes-function-calling-v1 conversation objects into unified JSON array records, converting all <tool_call> XML tags to raw JSON arrays."""
    import re

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    print(f"✍️ Formatting Hermes Function Calling records into unified JSON arrays to '{output_jsonl}'...", flush=True)

    total_records = 0
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for jpath in json_paths:
            with open(jpath, encoding="utf-8") as f_in:
                items = json.load(f_in)
                if not isinstance(items, list):
                    items = [items]

                for item in items:
                    conversations = item.get("conversations", [])
                    if not conversations:
                        continue

                    tools_str = ""
                    for turn in conversations:
                        r = turn.get("from") or turn.get("role") or ""
                        val = (turn.get("value") or turn.get("content") or "").strip()
                        if r == "system":
                            m = re.search(r"<tools>\s*(.*?)\s*</tools>", val, re.DOTALL)
                            if m:
                                tools_str = m.group(1).strip()

                    curr_query = ""
                    for turn in conversations:
                        r = turn.get("from") or turn.get("role") or ""
                        val = (turn.get("value") or turn.get("content") or "").strip()

                        if r in ["human", "user"]:
                            curr_query = val
                        elif r in ["gpt", "assistant"] and curr_query:
                            # Extract tool call XML blocks
                            tool_matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", val, re.DOTALL)
                            calls = []
                            if tool_matches:
                                for tm in tool_matches:
                                    try:
                                        calls.append(json.loads(tm))
                                    except Exception:
                                        pass
                            else:
                                try:
                                    parsed = json.loads(val)
                                    if isinstance(parsed, list):
                                        calls = parsed
                                    elif isinstance(parsed, dict):
                                        calls = [parsed]
                                except Exception:
                                    pass

                            if calls:
                                json_arr_str = json.dumps(calls, separators=(",", ":"))
                                if tools_str:
                                    prompt_text = (
                                        "Below is a user query along with available tool definitions in JSON format. "
                                        "Select the appropriate tool(s) and provide the tool call(s) as a strict JSON array.\n\n"
                                        f"<tools>\n{tools_str}\n</tools>\n\n"
                                        f"### Query:\n{curr_query}\n\n"
                                        f"### Response:\n{json_arr_str}"
                                    )
                                else:
                                    prompt_text = (
                                        "Below is a user query. Provide the appropriate tool call or structured JSON response.\n\n"
                                        f"### Query:\n{curr_query}\n\n"
                                        f"### Response:\n{json_arr_str}"
                                    )

                                record = json.dumps({"text": prompt_text})
                                f_out.write(record + "\n")
                                total_records += 1

    print(f"Formatted {total_records:,} unified JSON array instruction entries from Hermes into JSONL.", flush=True)
    return total_records


def prepare_hermes_fc(
    target_dir: str = "data/HermesFunctionCalling",
    min_count: int = 50,
    force: bool = False,
    vocab_map_in: str | None = None,
):
    """Download, format, tokenize, split, verify, and remediate Hermes Function Calling dataset."""
    target_dir = os.path.abspath(target_dir)
    raw_dir = os.path.join(target_dir, "raw")

    print("\n=== NousResearch Hermes Function Calling Dataset Pipeline ===")
    print(f"Target Directory: {target_dir}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=50_000, min_valid_tokens=5_000):
        print(f"[OK] Dataset 'HermesFunctionCalling' in '{target_dir}' verified and ready to consume! Skipping regeneration.", flush=True)
        return

    if force:
        print("Force flag specified. Re-generating Hermes FC dataset...", flush=True)
    else:
        print("Hermes FC dataset missing or corrupted. Triggering remediation...", flush=True)

    remediate_dataset(target_dir)

    # Step 1: Download JSON shards
    json_paths = download_hermes_function_calling_shards(raw_dir)

    # Step 2: Format records into JSONL
    formatted_jsonl = os.path.join(raw_dir, "formatted.jsonl")
    total_records = format_hermes_jsonl(json_paths, formatted_jsonl)
    if total_records == 0:
        raise RuntimeError("Formatted zero records from Hermes Function Calling dataset.")

    # Step 3: Tokenize using retokenize.py
    full_bin = os.path.join(target_dir, "full_trimmed.bin")
    vocab_map = os.path.join(target_dir, "vocab_map.json")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retokenize_script = os.path.join(project_root, "src", "retokenize.py")

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

    if vocab_map_in and os.path.exists(vocab_map_in):
        print(f"🔗 Aligning Hermes FC vocabulary with pretraining vocab map '{vocab_map_in}'...", flush=True)
        cmd.extend(["--vocab-map-in", vocab_map_in])
        shutil.copy(vocab_map_in, vocab_map)
    else:
        print("Generating vocabulary map for Hermes FC...", flush=True)
        cmd.extend(["--trim-vocab", "--min-count", str(min_count), "--vocab-map-out", vocab_map])

    print("\n🚀 Tokenizing Hermes FC formatted prompts using Gigatoken engine...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Retokenization failed for Hermes FC with exit code {res.returncode}")
    print(f"Tokenization complete in {time.time() - t0:.2f}s!", flush=True)

    full_data = np.memmap(full_bin, dtype=np.uint16, mode="r")
    total_tokens = len(full_data)

    # Step 4: Split into 95% train / 5% valid split
    train_tokens = int(total_tokens * 0.95)
    valid_tokens = total_tokens - train_tokens

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")

    train_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(train_tokens,))
    train_mmap[:] = full_data[:train_tokens]
    train_mmap.flush()
    del train_mmap

    valid_mmap = np.memmap(valid_bin, dtype=np.uint16, mode="w+", shape=(valid_tokens,))
    valid_mmap[:] = full_data[train_tokens : train_tokens + valid_tokens]
    valid_mmap.flush()
    del valid_mmap

    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)

    if os.path.exists(full_bin):
        os.remove(full_bin)
    if os.path.exists(formatted_jsonl):
        os.remove(formatted_jsonl)

    if not verify_dataset(target_dir, min_train_tokens=train_tokens, min_valid_tokens=valid_tokens):
        raise RuntimeError(f"Post-generation verification failed for Hermes FC in '{target_dir}'.")

    print("\n✅ Hermes Function Calling Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {train_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB, {valid_tokens:,} tokens)")


def prepare_router_blend(
    target_dir: str = "data/RouterBlend",
    synth_dir: str = "data/SynthAPIGen",
    hermes_dir: str = "data/HermesFunctionCalling",
    json_pretrain_dir: str = "data/JSONPretrain",
    target_train_tokens: int = 85_000_000,
    target_valid_tokens: int = 5_000_000,
    force: bool = False,
):
    """Combine 40% Synth-APIGen + 40% Hermes FC + 20% JSONPretrain (80/20 Anchor Blend) with block-level sequence shuffling."""
    target_dir = os.path.abspath(target_dir)
    print("\n=== Agentic Router 80/20 Anchor Curriculum Blend Pipeline ===")
    print(f"Target Directory: {target_dir}")
    print(f"Target Train Tokens: {target_train_tokens:,}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=50_000, min_valid_tokens=1_000):
        print(f"[OK] Dataset 'RouterBlend' in '{target_dir}' verified and ready to consume! Skipping regeneration.", flush=True)
        return

    # Ensure source datasets exist
    if not verify_dataset(synth_dir, min_train_tokens=10_000, min_valid_tokens=1_000):
        prepare_synth_apigen(target_dir=synth_dir, force=force)

    synth_vocab = os.path.join(synth_dir, "vocab_map.json")
    if not verify_dataset(hermes_dir, min_train_tokens=10_000, min_valid_tokens=1_000):
        prepare_hermes_fc(target_dir=hermes_dir, force=force, vocab_map_in=synth_vocab if os.path.exists(synth_vocab) else None)

    if not verify_dataset(json_pretrain_dir, min_train_tokens=10_000, min_valid_tokens=1_000):
        prepare_pretraining_dataset(
            target_dir=json_pretrain_dir,
            repo_id="HuggingFaceFW/fineweb-edu",
            dataset_name="FineWeb-Edu-JSON",
            target_tokens=2_600_000_000,
            valid_tokens=target_valid_tokens,
            force=force,
            vocab_map_in=synth_vocab if os.path.exists(synth_vocab) else None,
        )

    os.makedirs(target_dir, exist_ok=True)
    remediate_dataset(target_dir)

    synth_bin = os.path.join(synth_dir, "train_trimmed.bin")
    if not os.path.exists(synth_bin):
        synth_bin = os.path.join(synth_dir, "train.bin")

    hermes_bin = os.path.join(hermes_dir, "train_trimmed.bin")
    if not os.path.exists(hermes_bin):
        hermes_bin = os.path.join(hermes_dir, "train.bin")

    json_bin = os.path.join(json_pretrain_dir, "train_trimmed.bin")
    if not os.path.exists(json_bin):
        json_bin = os.path.join(json_pretrain_dir, "train.bin")

    # Copy vocabulary map
    target_vocab = os.path.join(target_dir, "vocab_map.json")
    if os.path.exists(synth_vocab):
        shutil.copy(synth_vocab, target_vocab)
    elif os.path.exists(os.path.join(json_pretrain_dir, "vocab_map.json")):
        shutil.copy(os.path.join(json_pretrain_dir, "vocab_map.json"), target_vocab)

    synth_mmap = np.memmap(synth_bin, dtype=np.uint16, mode="r")
    hermes_mmap = np.memmap(hermes_bin, dtype=np.uint16, mode="r")
    json_mmap = np.memmap(json_bin, dtype=np.uint16, mode="r")

    # Target 80/20 Anchor Blend: 40% SynthAPIGen, 40% HermesFC, 20% JSONPretrain
    req_synth = int(target_train_tokens * 0.40)
    req_hermes = int(target_train_tokens * 0.40)
    req_json = target_train_tokens - req_synth - req_hermes

    block_size = 2048
    synth_b_count = req_synth // block_size
    hermes_b_count = req_hermes // block_size
    json_b_count = req_json // block_size

    synth_max_b = len(synth_mmap) // block_size
    hermes_max_b = len(hermes_mmap) // block_size
    json_max_b = len(json_mmap) // block_size

    # Loop indices if target requirement exceeds source tokens
    blocks = [("synth", i % synth_max_b) for i in range(synth_b_count)] + [("hermes", i % hermes_max_b) for i in range(hermes_b_count)] + [("json", i % json_max_b) for i in range(json_b_count)]
    rng = np.random.default_rng(seed=42)
    rng.shuffle(blocks)

    total_blocks = len(blocks)
    actual_train_tokens = total_blocks * block_size

    print(f"Blending 40% Synth-APIGen ({synth_b_count * block_size:,} tokens) + 40% Hermes FC ({hermes_b_count * block_size:,} tokens) + 20% JSONPretrain ({json_b_count * block_size:,} tokens) = {actual_train_tokens:,} total tokens...", flush=True)

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    train_fallback = os.path.join(target_dir, "train.bin")

    out_mmap = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(actual_train_tokens,))

    chunk_blocks = 50_000
    for i in range(0, total_blocks, chunk_blocks):
        batch = blocks[i : i + chunk_blocks]
        out_offset = i * block_size

        for b_idx, (src, src_blk_idx) in enumerate(batch):
            if src == "synth":
                src_mmap = synth_mmap
            elif src == "hermes":
                src_mmap = hermes_mmap
            else:
                src_mmap = json_mmap

            src_start = src_blk_idx * block_size
            src_end = src_start + block_size
            dst_start = out_offset + (b_idx * block_size)
            dst_end = dst_start + block_size

            out_mmap[dst_start:dst_end] = src_mmap[src_start:src_end]

    out_mmap.flush()
    del out_mmap

    # Copy valid bin from Synth-APIGen
    synth_valid = os.path.join(synth_dir, "valid_trimmed.bin")
    if not os.path.exists(synth_valid):
        synth_valid = os.path.join(synth_dir, "valid.bin")

    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")
    if os.path.exists(synth_valid):
        shutil.copy(synth_valid, valid_bin)

    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)
    if not verify_dataset(target_dir, min_train_tokens=actual_train_tokens, min_valid_tokens=1_000):
        raise RuntimeError(f"Post-generation verification failed for RouterBlend in '{target_dir}'.")

    print("\n✅ Router 80/20 Anchor Blend Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {actual_train_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB)")


def download_tinystories_shards(raw_dir: str) -> tuple[str, str]:
    """Download roneneldan/TinyStories dataset text files."""
    os.makedirs(raw_dir, exist_ok=True)
    train_txt = os.path.join(raw_dir, "TinyStories-train.txt")
    valid_txt = os.path.join(raw_dir, "TinyStories-valid.txt")

    print("🔍 Downloading roneneldan/TinyStories dataset files...", flush=True)
    if not os.path.exists(train_txt) or os.path.getsize(train_txt) == 0:
        train_txt = hf_hub_download(
            repo_id="roneneldan/TinyStories",
            filename="TinyStories-train.txt",
            repo_type="dataset",
            local_dir=raw_dir,
        )
    else:
        print(f"  - Already exists: {train_txt} ({os.path.getsize(train_txt) / (1024**2):.1f} MB)", flush=True)

    if not os.path.exists(valid_txt) or os.path.getsize(valid_txt) == 0:
        valid_txt = hf_hub_download(
            repo_id="roneneldan/TinyStories",
            filename="TinyStories-valid.txt",
            repo_type="dataset",
            local_dir=raw_dir,
        )
    else:
        print(f"  - Already exists: {valid_txt} ({os.path.getsize(valid_txt) / (1024**2):.1f} MB)", flush=True)

    return train_txt, valid_txt


def prepare_tinystories(
    target_dir: str = "data/TinyStories",
    min_count: int = 50,
    force: bool = False,
    vocab_map_in: str | None = None,
):
    """Download, tokenize, verify, and remediate roneneldan/TinyStories dataset."""
    target_dir = os.path.abspath(target_dir)
    raw_dir = os.path.join(target_dir, "raw")

    print("\n=== roneneldan/TinyStories Pretraining Dataset Pipeline ===")
    print(f"Target Directory: {target_dir}\n")

    if not force and verify_dataset(target_dir, min_train_tokens=100_000, min_valid_tokens=10_000):
        print(f"[OK] Dataset 'TinyStories' in '{target_dir}' verified and ready to consume! Skipping regeneration.", flush=True)
        return

    if force:
        print("Force flag specified. Re-generating TinyStories dataset...", flush=True)
    else:
        print("TinyStories dataset missing or corrupted. Triggering remediation...", flush=True)

    remediate_dataset(target_dir)

    # Step 1: Download text files
    train_txt, valid_txt = download_tinystories_shards(raw_dir)

    # Step 2: Tokenize using retokenize.py
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    retokenize_script = os.path.join(project_root, "src", "retokenize.py")
    vocab_map = os.path.join(target_dir, "vocab_map.json")

    train_bin = os.path.join(target_dir, "train_trimmed.bin")
    valid_bin = os.path.join(target_dir, "valid_trimmed.bin")

    cmd_train = [
        sys.executable,
        retokenize_script,
        "-i",
        train_txt,
        "-o",
        train_bin,
        "--file-type",
        "text",
    ]
    if vocab_map_in and os.path.exists(vocab_map_in):
        print(f"🔗 Aligning TinyStories train vocabulary with pretraining vocab map '{vocab_map_in}'...", flush=True)
        cmd_train.extend(["--vocab-map-in", vocab_map_in])
        shutil.copy(vocab_map_in, vocab_map)
    else:
        print("Generating vocabulary map for TinyStories...", flush=True)
        cmd_train.extend(["--trim-vocab", "--min-count", str(min_count), "--vocab-map-out", vocab_map])

    print("\n🚀 Tokenizing TinyStories train set using Gigatoken engine...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_train, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Retokenization failed for TinyStories train set with exit code {res.returncode}")
    print(f"Train tokenization complete in {time.time() - t0:.2f}s!", flush=True)

    cmd_valid = [
        sys.executable,
        retokenize_script,
        "-i",
        valid_txt,
        "-o",
        valid_bin,
        "--file-type",
        "text",
        "--vocab-map-in",
        vocab_map,
    ]
    print("\n🚀 Tokenizing TinyStories valid set using Gigatoken engine...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_valid, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Retokenization failed for TinyStories valid set with exit code {res.returncode}")
    print(f"Valid tokenization complete in {time.time() - t0:.2f}s!", flush=True)

    train_fallback = os.path.join(target_dir, "train.bin")
    valid_fallback = os.path.join(target_dir, "valid.bin")

    if os.path.exists(train_fallback) or os.path.islink(train_fallback):
        os.remove(train_fallback)
    os.symlink("train_trimmed.bin", train_fallback) if hasattr(os, "symlink") else shutil.copy(train_bin, train_fallback)

    if os.path.exists(valid_fallback) or os.path.islink(valid_fallback):
        os.remove(valid_fallback)
    os.symlink("valid_trimmed.bin", valid_fallback) if hasattr(os, "symlink") else shutil.copy(valid_bin, valid_fallback)

    if not verify_dataset(target_dir, min_train_tokens=100_000, min_valid_tokens=10_000):
        raise RuntimeError(f"Post-generation verification failed for TinyStories in '{target_dir}'.")

    train_tokens = len(np.memmap(train_bin, dtype=np.uint16, mode="r"))
    valid_tokens = len(np.memmap(valid_bin, dtype=np.uint16, mode="r"))

    print("\n✅ TinyStories Preparation Complete!")
    print(f"  - Train Dataset: '{train_bin}' ({os.path.getsize(train_bin) / (1024**2):.2f} MB, {train_tokens:,} tokens)")
    print(f"  - Valid Dataset: '{valid_bin}' ({os.path.getsize(valid_bin) / (1024**2):.2f} MB, {valid_tokens:,} tokens)")
    print(f"  - Vocab Map:     '{vocab_map}'")


def main():
    parser = argparse.ArgumentParser(description="Generalized Dataset Preparation Pipeline for Pretraining & Finetuning.")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=[
            "all",
            "tinystories",
            "fineweb",
            "fineweb-edu",
            "cosmopedia",
            "cosmopedia-v2",
            "bookcorpus",
            "bookcorpus-fineweb-edu",
            "platypus",
            "open-platypus",
            "synth-apigen",
            "hermes-fc",
            "json-pretrain",
            "router-blend",
        ],
        default="all",
        help="Dataset(s) to download and prepare (default: all)",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Target root directory override for dataset output")
    parser.add_argument("--tinystories-dir", type=str, default="data/TinyStories", help="Directory for TinyStories pretraining dataset")
    parser.add_argument("--fineweb-dir", type=str, default="data/FineWeb", help="Directory for FineWeb pretraining dataset")
    parser.add_argument("--fineweb-edu-dir", type=str, default="data/FineWebEdu", help="Directory for FineWeb-Edu pretraining dataset")
    parser.add_argument("--cosmopedia-dir", type=str, default="data/CosmopediaV2", help="Directory for Cosmopedia v2 pretraining dataset")
    parser.add_argument("--bookcorpus-dir", type=str, default="data/BookCorpus", help="Directory for BookCorpus pretraining dataset")
    parser.add_argument("--blend-dir", type=str, default="data/BookCorpusFineWebEdu", help="Directory for custom BookCorpus+FineWeb-Edu blend dataset")
    parser.add_argument("--platypus-dir", type=str, default="data/OpenPlatypus", help="Directory for Open-Platypus dataset")
    parser.add_argument("--synth-dir", type=str, default="data/SynthAPIGen", help="Directory for Argilla Synth-APIGen tool-calling dataset")
    parser.add_argument("--hermes-dir", type=str, default="data/HermesFunctionCalling", help="Directory for NousResearch Hermes Function Calling dataset")
    parser.add_argument("--json-pretrain-dir", type=str, default="data/JSONPretrain", help="Directory for Structured JSON pretraining dataset")
    parser.add_argument("--router-blend-dir", type=str, default="data/RouterBlend", help="Directory for Agentic Router curriculum blend dataset")
    parser.add_argument("--target-tokens", type=int, default=2_600_000_000, help="Target pretraining train token count (default: 2.6B)")
    parser.add_argument("--valid-tokens", type=int, default=5_000_000, help="Target pretraining valid token count (default: 5M)")
    parser.add_argument("--min-count", type=int, default=50, help="Vocabulary trimming minimum count threshold")
    parser.add_argument("--vocab-map-in", type=str, default=None, help="Align dataset vocabulary with existing pretraining vocab map")
    parser.add_argument("--force", action="store_true", default=False, help="Force dataset re-download and re-tokenization")

    args = parser.parse_args()

    dataset_choice = args.dataset.lower()

    tinystories_path = args.data_dir if (args.data_dir and dataset_choice == "tinystories") else args.tinystories_dir
    fineweb_path = args.data_dir if (args.data_dir and dataset_choice == "fineweb") else args.fineweb_dir
    fineweb_edu_path = args.data_dir if (args.data_dir and dataset_choice == "fineweb-edu") else args.fineweb_edu_dir
    cosmopedia_path = args.data_dir if (args.data_dir and dataset_choice in ["cosmopedia", "cosmopedia-v2"]) else args.cosmopedia_dir
    bookcorpus_path = args.data_dir if (args.data_dir and dataset_choice == "bookcorpus") else args.bookcorpus_dir
    blend_path = args.data_dir if (args.data_dir and dataset_choice in ["bookcorpus-fineweb-edu", "blend"]) else args.blend_dir
    platypus_path = args.data_dir if (args.data_dir and dataset_choice in ["platypus", "open-platypus"]) else args.platypus_dir
    synth_path = args.data_dir if (args.data_dir and dataset_choice == "synth-apigen") else args.synth_dir
    hermes_path = args.data_dir if (args.data_dir and dataset_choice == "hermes-fc") else args.hermes_dir
    json_pretrain_path = args.data_dir if (args.data_dir and dataset_choice == "json-pretrain") else args.json_pretrain_dir
    router_blend_path = args.data_dir if (args.data_dir and dataset_choice == "router-blend") else args.router_blend_dir

    if dataset_choice in ["all", "tinystories"]:
        prepare_tinystories(
            target_dir=tinystories_path,
            min_count=args.min_count,
            force=args.force,
            vocab_map_in=args.vocab_map_in,
        )

    if dataset_choice in ["all", "fineweb"]:
        prepare_pretraining_dataset(
            target_dir=fineweb_path,
            repo_id="HuggingFaceFW/fineweb",
            dataset_name="FineWeb",
            target_tokens=args.target_tokens,
            valid_tokens=args.valid_tokens,
            min_count=args.min_count,
            force=args.force,
            vocab_map_in=args.vocab_map_in,
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
            vocab_map_in=args.vocab_map_in,
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

    if dataset_choice in ["all", "bookcorpus"]:
        prepare_pretraining_dataset(
            target_dir=bookcorpus_path,
            repo_id="lucadiliello/bookcorpusopen",
            dataset_name="BookCorpus",
            target_tokens=args.target_tokens,
            valid_tokens=args.valid_tokens,
            min_count=args.min_count,
            force=args.force,
        )

    if dataset_choice in ["all", "bookcorpus-fineweb-edu", "blend"]:
        prepare_custom_blend(
            target_dir=blend_path,
            bookcorpus_dir=bookcorpus_path,
            fineweb_edu_dir=fineweb_edu_path,
            target_train_tokens=args.target_tokens,
            target_valid_tokens=args.valid_tokens,
            force=args.force,
        )

    if dataset_choice in ["all", "platypus", "open-platypus"]:
        prepare_open_platypus(
            target_dir=platypus_path,
            min_count=args.min_count,
            force=args.force,
            fineweb_dir=fineweb_edu_path if os.path.exists(fineweb_edu_path) else fineweb_path,
        )

    if dataset_choice in ["all", "synth-apigen"]:
        prepare_synth_apigen(
            target_dir=synth_path,
            min_count=args.min_count,
            force=args.force,
            vocab_map_in=args.vocab_map_in,
        )

    if dataset_choice in ["all", "hermes-fc"]:
        prepare_hermes_fc(
            target_dir=hermes_path,
            min_count=args.min_count,
            force=args.force,
            vocab_map_in=args.vocab_map_in or (os.path.join(synth_path, "vocab_map.json") if os.path.exists(os.path.join(synth_path, "vocab_map.json")) else None),
        )

    if dataset_choice in ["all", "json-pretrain"]:
        prepare_pretraining_dataset(
            target_dir=json_pretrain_path,
            repo_id="HuggingFaceFW/fineweb-edu",
            dataset_name="FineWeb-Edu-JSON",
            target_tokens=args.target_tokens,
            valid_tokens=args.valid_tokens,
            min_count=args.min_count,
            force=args.force,
            vocab_map_in=args.vocab_map_in,
        )

    if dataset_choice in ["all", "router-blend"]:
        prepare_router_blend(
            target_dir=router_blend_path,
            synth_dir=synth_path,
            hermes_dir=hermes_path,
            json_pretrain_dir=json_pretrain_path,
            target_train_tokens=args.target_tokens,
            target_valid_tokens=args.valid_tokens,
            force=args.force,
        )


if __name__ == "__main__":
    main()
