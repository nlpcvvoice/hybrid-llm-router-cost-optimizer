#!/usr/bin/env python3
"""Task 1.1: Extract seed prompts from LMSYS parquet files.

Reads LMSYS-Chat parquet files, takes the FIRST user message from each
conversation, cleans it, and writes N seed prompts to a JSONL file.

Seed prompt = a real user question asked to a large model, used later as
raw material for the router difficulty-labelling dataset.
"""
import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def extract_first_user(content: str) -> str | None:
    """Return first non-empty consecutive user utterance, cleaned."""
    if not content:
        return None
    text = content.strip()
    if not text:
        return None
    return text


def clean(text: str, min_len: int = 10, max_len: int = 4000) -> str | None:
    """Return cleaned text or None if it should be dropped."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_len:
        return None
    if len(text) > max_len:
        return None
    # Drop prompts that are mostly non-ascii (header/artifacts) or empty-noise
    if len(text) < 2:
        return None
    return text


def load_seed_prompts(parquet_paths, limit) -> list:
    seen = set()
    seeds = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rgi in range(pf.num_row_groups):
            if len(seeds) >= limit:
                return seeds[:limit]
            df = pf.read_row_group(rgi).to_pandas()
            for _, row in df.iterrows():
                if len(seeds) >= limit:
                    return seeds[:limit]
                for msg in row["conversation"]:
                    if msg.get("role") == "user":
                        raw = extract_first_user(msg.get("content", ""))
                        if raw is None:
                            break
                        cleaned = clean(raw)
                        if cleaned is None:
                            break  # skip this conversation's noisy user msg
                        if cleaned not in seen:
                            seen.add(cleaned)
                            seeds.append(cleaned)
                        break
    return seeds[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True,
                    help="LMSYS parquet file(s)")
    ap.add_argument("--output", default="data/seed_prompts.jsonl")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    seeds = load_seed_prompts(args.parquet, args.limit)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps({"prompt": s, "source": "lmsys"}) + "\n")
    print(f"[seed] wrote {len(seeds)} seed prompts -> {out}")


if __name__ == "__main__":
    main()
