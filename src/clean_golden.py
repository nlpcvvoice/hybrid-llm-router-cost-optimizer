#!/usr/bin/env python3
"""Task 1.4: Validate & clean the golden router dataset.

Reads the raw labeled output from `make_router_dataset.py`, applies a final
layer of strict validation/dedup, flags low-confidence records, and writes a
clean golden set used to SFT-tune the 1.5B router. Also reports class balance
(LOW vs HIGH), which matters for training.
"""
import argparse
import json
import statistics
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from typing import Literal


class Golden(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=4000)
    difficulty: Literal["LOW", "HIGH"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=120)
    target: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/router_train_gold.jsonl")
    ap.add_argument("--output", default="data/router_train_gold_clean.jsonl")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="drop records below this confidence")
    args = ap.parse_args()

    clean, bad = [], 0
    seen = set()
    for line in open(args.input, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rec = Golden.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as e:
            bad += 1
            continue
        if rec.confidence_score < args.conf_threshold:
            bad += 1
            continue
        if rec.prompt in seen:
            continue  # dedup
        seen.add(rec.prompt)
        clean.append(rec)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in clean:
            f.write(c.model_dump_json(ensure_ascii=False) + "\n")

    low = sum(1 for c in clean if c.difficulty == "LOW")
    high = len(clean) - low
    confs = [c.confidence_score for c in clean]
    print(f"[clean] {len(clean)} clean records (dropped {bad})")
    print(f"[clean] LOW={low} HIGH={high}  ratio={low/max(high,1):.1f}:1")
    print(f"[clean] confidence mean={statistics.mean(confs):.3f} "
          f"min={min(confs):.2f} max={max(confs):.2f}")
    print(f"[clean] wrote -> {out}")


if __name__ == "__main__":
    main()
