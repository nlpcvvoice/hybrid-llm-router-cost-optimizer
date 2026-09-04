#!/usr/bin/env python3
"""Task 1.2 + 1.4: Label seed prompts as [LOW]/[HIGH] via OpenRouter.

Reads `seed_prompts.jsonl`, asks an OpenRouter free model to classify each
prompt's difficulty, validates the output with Pydantic, and writes a clean
golden router dataset used later to SFT-tune the 1.5B router.

Task 1.3 (empirical calibration with a local 7B) is applied separately.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from openrouter_client import OpenRouterClient

SYSTEM_PROMPT = (
    "You are an expert AI System Architect specializing in LLM cost "
    "optimization. Classify whether the user prompt can be solved by a "
    "lightweight 7B-parameter local model (LOW) or requires a frontier model "
    "like GPT-4o (HIGH). "
    "Return ONLY a JSON object: "
    '{"difficulty":"LOW|HIGH","confidence_score":0.0-1.0,"reasoning":"<=60 chars"}.'
)


class RouteAnnotation(BaseModel):
    difficulty: Literal["LOW", "HIGH"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=120)


def parse_annotation(raw: str) -> RouteAnnotation | None:
    """Parse model output into a RouteAnnotation; drop invalid records."""
    row = None
    for candidate in (raw, raw.strip("`"), raw.strip()):
        try:
            row = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    if row is None:
        return None
    try:
        return RouteAnnotation.model_validate(row)
    except ValidationError:
        return None


def annotate(client: OpenRouterClient, prompt: str) -> dict | None:
    try:
        raw = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            json_mode=True,
            max_tokens=120,
        )
    except Exception as e:
        print(f"[annotate] error key_set={client.key_set}: {type(e).__name__}")
        return None
    ann = parse_annotation(raw)
    if ann is None:
        return None
    return {
        "prompt": prompt,
        "target": f"[{ann.difficulty}]",
        "difficulty": ann.difficulty,
        "confidence_score": ann.confidence_score,
        "reasoning": ann.reasoning,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/seed_prompts.jsonl")
    ap.add_argument("--output", default="data/router_train_gold.jsonl")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrency; keep 1 to avoid free-model rate limits")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        prompts = [json.loads(l)["prompt"] for l in f if l.strip()]
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: args.limit]

    client = OpenRouterClient()
    print(f"[make_router] key_set={client.key_set} n={len(prompts)} "
          f"workers={args.workers}")
    if not client.key_set:
        print("[make_router] FATAL: no API key; abort")
        return

    gold, skipped = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(annotate, client, p): p for p in prompts}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res:
                gold.append(res)
            else:
                skipped += 1
            if i % 200 == 0:
                print(f"[make_router] {i}/{len(prompts)} done, "
                      f"kept={len(gold)} skipped={skipped}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    low = sum(1 for g in gold if g["difficulty"] == "LOW")
    print(f"[make_router] DONE: kept={len(gold)} skipped={skipped} "
          f"LOW={low} HIGH={len(gold)-low}")
    print(f"[make_router] wrote -> {out}")


if __name__ == "__main__":
    main()
