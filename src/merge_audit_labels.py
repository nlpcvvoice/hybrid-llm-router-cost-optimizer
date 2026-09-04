#!/usr/bin/env python3
"""Merge Gemini audit into the golden set (final audited labels).

Task 1.3b: gemini-2.5-flash audited all golden-set HIGH prompts and decided
either KEEP_HIGH or DOWNGRADE_LOW. This combines that verdict with the
original labels to produce the FINAL audited golden dataset:
  - original LOW  -> LOW (unchanged)
  - original HIGH + audit KEEP_HIGH    -> HIGH
  - original HIGH + audit DOWNGRADE_LOW -> LOW  (was downgraded)
Writes router_train_gold_audited.jsonl and prints class stats.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/router_train_gold_clean.jsonl")
    ap.add_argument("--audit", default="data/gemini_audit_high.jsonl")
    ap.add_argument("--output", default="data/router_train_gold_audited.jsonl")
    args = ap.parse_args()

    golden = [json.loads(l) for l in open(args.golden, encoding="utf-8") if l.strip()]
    audit = {}
    for l in open(args.audit, encoding="utf-8"):
        d = json.loads(l)
        audit[d["prompt"]] = d["audit"]["audit_verdict"]

    out = []
    for r in golden:
        row = dict(r)
        if r["difficulty"] == "HIGH":
            v = audit.get(r["prompt"])
            if v == "KEEP_HIGH":
                row["audited_label"] = "HIGH"
            elif v == "DOWNGRADE_LOW":
                row["audited_label"] = "LOW"
            else:
                row["audited_label"] = "HIGH"  # untested HIGH stays HIGH
        else:
            row["audited_label"] = "LOW"
        out.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    c = Counter(x["audited_label"] for x in out)
    orig = Counter(x["difficulty"] for x in out)
    downgraded = sum(
        1 for r in out
        if r["difficulty"] == "HIGH" and r["audited_label"] == "LOW"
    )
    kept = sum(
        1 for r in out
        if r["difficulty"] == "HIGH" and r["audited_label"] == "HIGH"
    )
    print(f"[merge] total={len(out)}")
    print(f"[merge] original: {dict(orig)}")
    print(f"[merge] audited : {dict(c)}")
    print(f"[merge] HIGH kept={kept} downgraded_to_LOW={downgraded}")
    print(f"[merge] wrote -> {args.output}")


if __name__ == "__main__":
    main()