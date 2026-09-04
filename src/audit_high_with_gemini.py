#!/usr/bin/env python3
"""Phase 1.3b: Audit HIGH-label samples with Gemini 2.5 Flash (no manual review).

Task 1.3 showed HIGH labels are only ~30% stable under re-labeling. Per user
decision, an automated auditor (gemini-2.5-flash on Vertex AI, free tier)
reviews every golden-set HIGH prompt and decides:
  KEEP_HIGH     -> genuinely needs a frontier model
  DOWNGRADE_LOW -> a local 7B can handle it (over-conservative label)
Each call: max_output_tokens <= 1024, serial (1 worker), single sanity call
was verified before this run. Stops on 429/quota and reports metrics.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from vertexai.generative_models import GenerativeModel

SYSTEM_PROMPT = (
    "You are an independent LLM-router QA auditor. Decide whether a user prompt "
    "truly REQUIRES a frontier/expensive cloud model (label=KEEP_HIGH), or "
    "whether a lightweight 7B-parameter local model can answer it well "
    "(label=DOWNGRADE_LOW). Consider: knowledge depth, reasoning length, "
    "long-context needs, code, math, and multi-step planning. "
    "Return ONLY a JSON object with exactly these keys: "
    '{"audit_verdict":"KEEP_HIGH|DOWNGRADE_LOW","confidence":0.0-1.0,'
    '"reason":"<=60 chars"}'
)


class Audit(BaseModel):
    audit_verdict: Literal["KEEP_HIGH", "DOWNGRADE_LOW"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., max_length=120)


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object out of Gemini output (handles ```json fences)."""
    if not text:
        return None
    text = text.strip()
    # strip markdown code fence if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text)
    # find the outermost {...} span
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/router_train_gold_clean.jsonl")
    ap.add_argument("--output", default="data/gemini_audit_high.jsonl")
    ap.add_argument("--limit", type=int, default=198,
                    help="cap HIGH samples to audit (free-tier budget)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--resume", action="store_true",
                    help="skip prompts already present in --output")
    args = ap.parse_args()

    import vertexai
    vertexai.init()
    model = GenerativeModel(args.model, system_instruction=[SYSTEM_PROMPT])

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    high = [r for r in rows if r["difficulty"] == "HIGH"][: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    done_prompts = set()
    if args.resume and out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done_prompts.add(json.loads(line)["prompt"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [r for r in high if r["prompt"] not in done_prompts]
    print(f"[audit] model={args.model} total_high={len(high)} "
          f"done={len(done_prompts)} todo={len(todo)} auth via VM service account",
          flush=True)

    f = open(out, "a", encoding="utf-8")
    errors = 0
    keep = 0
    for i, r in enumerate(todo, 1):
        try:
            resp = model.generate_content(
                r["prompt"],
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1024,
                },
            )
            parsed = None
            obj = _extract_json(resp.text)
            if obj is not None:
                try:
                    parsed = Audit.model_validate(obj)
                except ValidationError:
                    parsed = None
            if parsed is None:
                errors += 1
                print(f"[audit] {i}/{len(todo)} PARSE_ERR, treating as DOWNGRADE_LOW",
                      flush=True)
                parsed = Audit(audit_verdict="DOWNGRADE_LOW",
                               confidence=0.5, reason="unparsable")
            rec = {
                "prompt": r["prompt"],
                "orig_confidence": r["confidence_score"],
                "orig_reasoning": r["reasoning"],
                "audit": parsed.model_dump(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if parsed.audit_verdict == "KEEP_HIGH":
                keep += 1
        except Exception as e:
            # 429/quota -> STOP and report per Vertex guide
            print(f"[audit] ERROR at {i}: type={type(e).__name__} "
                  f"msg={str(e)[:120]}", flush=True)
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("[audit] QUOTA hit - stopping (free allowance). "
                      f"wrote={i - 1}", flush=True)
                break
            errors += 1

        if i % 10 == 0 or i == len(todo):
            print(f"[audit] {i}/{len(todo)} done, keep_high={keep} "
                  f"errors={errors}", flush=True)
    f.close()

    total = sum(
        1 for _ in open(out, encoding="utf-8") if _.strip()
    ) if out.exists() else 0
    downgrade = total - keep
    print(f"[audit] DONE: total={total} KEEP_HIGH={keep} "
          f"DOWNGRADE_LOW={downgrade} errors={errors}")
    print(f"[audit] wrote -> {out}")


if __name__ == "__main__":
    main()