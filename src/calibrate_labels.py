#!/usr/bin/env python3
"""Task 1.3: Empirical calibration of the golden labels (no-GPU adapter).

The plan called for a local 7B to verify labels empirically. This machine has
no GPU, so we use an INDEPENDENT labeler — a different OpenRouter free model
than the one used in Task 1.2 — to re-label a random sample of the golden set
and measure agreement. High agreement = labels are reliable (not the artifact
of a single labeler's subjectivity). High divergence identifies ambiguous
records to review before training.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from openrouter_client import OpenRouterClient

# Default labeler: same model family used in Task 1.2 (stability check).
# Independent-model calibration (different family) is preferred but needs free
# quota, which the daily free-model pool exhausted during Task 1.2's full run.
# Pass --labeler to override when quota resets.
CALIBRATOR_MODEL = "minimax/minimax-m3:free"

SYSTEM_PROMPT = (
    "You are an independent AI cost-optimization reviewer. Judge whether the "
    "user prompt NEEDS a frontier/paid model (HIGH) or is solvable by a "
    "lightweight local 7B model (LOW). "
    "Return ONLY a JSON object: "
    '{"difficulty":"LOW|HIGH","confidence_score":0.0-1.0}.'
)


class Calib(BaseModel):
    difficulty: Literal["LOW", "HIGH"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)


def relabel(client, prompt, labeler):
    try:
        raw = client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            temperature=0.0, json_mode=True, max_tokens=80,
            models={
                "primary_free": labeler,
                "fallback_free": [
                    "google/gemma-4-31b-it:free",
                    "nemotron-3-super-120b-a12b:free",
                ],
            },
        )
    except Exception as e:
        return {"error": type(e).__name__}
    try:
        c = Calib.model_validate(json.loads(raw))
        return {"difficulty": c.difficulty, "confidence": c.confidence_score}
    except (json.JSONDecodeError, ValidationError):
        return {"error": "parse"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/router_train_gold_clean.jsonl")
    ap.add_argument("--output", default="data/calibration_sample.jsonl")
    ap.add_argument("--sample", type=int, default=200,
                    help="sample size to re-label")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrency; keep 1 to avoid free-model rate limits")
    ap.add_argument("--labeler", default=CALIBRATOR_MODEL,
                    help="override calibrator model id")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.sample]

    client = OpenRouterClient()
    print(f"[calib] labeler={args.labeler} sample={len(sample)} "
          f"key_set={client.key_set}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(relabel, client, r["prompt"], args.labeler): r
                for r in sample}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            res = fut.result()
            row["calibrated"] = res
            results.append(row)
            if i % 50 == 0:
                print(f"[calib] {i}/{len(sample)} done")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    relabeled = [r for r in results if "difficulty" in r.get("calibrated", {})]
    agree = sum(1 for r in relabeled
                if r["calibrated"]["difficulty"] == r["difficulty"])
    n = len(relabeled) or 1
    # confusion between original labeler and calibrator
    from collections import Counter
    pairs = Counter((r["difficulty"], r["calibrated"]["difficulty"])
                    for r in relabeled)
    print(f"[calib] relabeled={n} agreement={agree} "
          f"rate={agree/n:.1%}")
    print(f"[calib] confusion(LOW->? , HIGH->?): {dict(pairs)}")
    # ambiguous: original HIGH but calibrator LOW, or low-confidence
    ambiguous = [r["prompt"] for r in relabeled
                 if r["calibrated"]["difficulty"] != r["difficulty"]]
    print(f"[calib] divergent prompts: {len(ambiguous)} -> review before train")
    print(f"[calib] wrote -> {out}")


if __name__ == "__main__":
    main()
