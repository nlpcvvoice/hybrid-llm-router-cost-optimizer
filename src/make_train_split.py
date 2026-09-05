#!/usr/bin/env python3
"""Stratified train/val/test split + class-balance for the audited gold set.

P2 data prep (host-agnostic):
  1. Stratify on the FINAL label (audited_label) so every split keeps both classes.
  2. Under-sample LOW in the TRAIN split only (target LOW:HIGH ratio),
     because val/test must reflect the real-world imbalanced distribution.
  3. Class *weighting* for the loss is done later in the training layer (P2),
     NOT here.

Writes data/train.jsonl, data/val.jsonl, data/test.jsonl and prints class stats.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def dump_rows(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/router_train_gold_audited.jsonl")
    ap.add_argument("--outdir", default="data", help="top-level so gitignore applies")
    ap.add_argument("--split", default="90,5,5", help="train,val,test percentages")
    ap.add_argument("--undersample", type=float, default=3.0,
                    help="target LOW:HIGH ratio inside TRAIN only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-key", default="audited_label")
    args = ap.parse_args()

    pct_train, pct_val, pct_test = (float(x) for x in args.split.split(","))
    assert abs(pct_train + pct_val + pct_test - 100.0) < 1e-6

    rows = load(args.input)
    key = args.label_key
    by_label = {}
    for r in rows:
        by_label.setdefault(r[key], []).append(r)

    rng = random.Random(args.seed)
    train, valid, test = [], [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_val = int(round(n * pct_val / 100))
        n_test = int(round(n * pct_test / 100))
        train.extend(group[n_val + n_test:])
        valid.extend(group[:n_val])
        test.extend(group[n_val:n_val + n_test])

    tr_counter = Counter(r[key] for r in train)
    if args.undersample is not None and "LOW" in tr_counter and "HIGH" in tr_counter:
        cap = int(tr_counter["HIGH"] * args.undersample)
        if tr_counter["LOW"] > cap:
            low = [r for r in train if r[key] == "LOW"]
            rng.shuffle(low)
            dropped = tr_counter["LOW"] - cap
            kept_low = low[:cap]
            train = [r for r in train if r[key] == "HIGH"] + kept_low
            print(f"[undersample] LOW {tr_counter['LOW']} -> {cap} (dropped {dropped})")
            print(f"              ratio LOW:HIGH = {args.undersample}:1")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    dump_rows(train, f"{args.outdir}/train.jsonl")
    dump_rows(valid, f"{args.outdir}/val.jsonl")
    dump_rows(test, f"{args.outdir}/test.jsonl")

    for name, rows_ in (("train", train), ("val", valid), ("test", test)):
        c = Counter(r[key] for r in rows_)
        print(f"[{name}] n={len(rows_)}  {dict(c)}  "
              f"ratio={c.get('LOW', 0) / max(c.get('HIGH', 0), 1):.1f}:1")


if __name__ == "__main__":
    main()