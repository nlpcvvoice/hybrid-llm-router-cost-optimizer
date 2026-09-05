# Hybrid LLM Router & Cost Optimizer

[![CI](https://img.shields.io/github/actions/workflow/status/nlpcvvoice/hybrid-llm-router-cost-optimizer/ci.yml?branch=main&logo=github)](https://github.com/nlpcvvoice/hybrid-llm-router-cost-optimizer/actions/workflows/ci.yml)

A smart routing + cost-optimization gateway for LLM APIs: fine-tune a small routing model (Qwen-1.5B) to judge request difficulty at a glance, send simple requests to a local 7B model and complex ones to an OpenRouter free model, with cascading fallback and FinOps billing analytics. Built entirely with free models; the interface is extensible to any paid model later.

## Status

- Phase 1 (data engineering) **complete**: seed → labeled gold → cleaned → calibrated → audited golden set
- Phase 2 (fine-tuning) **in progress**: stratified split + Colab LoRA trainer ready, local CPU smoke next
- CI **passing** (syntax + notebook validation on every push/PR)

> 🎬 Live demo: coming with the Phase 3 gateway (router forwards LOW → local tier, HIGH → cloud pool).

## Overview

| Tier | Executing model | Location | Cost |
|------|----------------|----------|------|
| Routing decision | Router-1.5B (fine-tuned) | Local GPU0 | $0 |
| [LOW] simple request | Local-7B-AWQ | Local GPU1 | $0 |
| [HIGH] hard request | OpenRouter free model (default `minimax/minimax-m3:free`) | Cloud API | $0 |
| [HIGH] extension slot | Any paid model (e.g. GPT-4o) | Cloud API | Enterprise config |

## Directory Structure

```
src/                    Core code
  openrouter_client.py   OpenRouter free-model client + auto-rotation + paid extension slot
  make_seed_dataset.py   Task 1.1: extract seed prompts from LMSYS parquet
  make_router_dataset.py Task 1.2/1.4: label difficulty via OpenRouter + Pydantic validation
configs/                Model list / billing config
data/                   Seed data / golden dataset (raw files git-ignored)
scripts/                Startup scripts
tests/                  Tests
```

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt   # (P0: requirements.txt to be added)
```

Load the seed prompts into `data/seed_prompts.jsonl`:

```bash
python src/make_seed_dataset.py \
  --parquet data/train-00000-of-00006-*.parquet data/train-00001-of-00006-*.parquet \
  --output data/seed_prompts.jsonl \
  --limit 2000
```

Label the difficulty of each seed prompt via OpenRouter free models:

```bash
python src/make_router_dataset.py \
  --input data/seed_prompts.jsonl \
  --output data/router_train_gold.jsonl \
  --limit 2000
```

> The OpenRouter API key is read from a shared `.env` at runtime (memory only, never printed, logged, or committed). Only `key_set=True/False` is ever logged.

## Skills Used

Accumulated per phase (updated as the project progresses).

| Skill | Evidence | Phase |
|-------|----------|-------|
| Data engineering pipeline (parquet → jsonl, dedup, seeding) | `src/make_seed_dataset.py` | P1 |
| LLM API reliability (free-model pool, 429 backoff, auto-rotation) | `src/openrouter_client.py` | P1 |
| Prompt engineering + structured output (Pydantic-validated labels) | `src/make_router_dataset.py` | P1 |
| Human⇄model calibration & imbalance-aware per-class reporting | `src/calibrate_labels.py` | P1 |
| LLM-as-judge auditing of low-consensus labels | `src/audit_high_with_gemini.py` | P1 |
| Stratified split + undersampling for class balance (3:1) | `src/make_train_split.py` | P2 |
| LoRA fine-tuning (fp16, no bitsandbytes) + class-weighted loss | `src/colab_train_classifier_head.py` | P2 |
| Self-contained Colab notebook engineering (buildable + compilable) | `src/build_colab_notebook.py` | P2 |
| CI/CD: GitHub Actions syntax + notebook validation | `.github/workflows/ci.yml` | P2 |

## Roadmap

| Phase | Scope |
|-------|-------|
| P1 | Data engineering: seed extraction, labeling, calibration |
| P2 | SFT fine-tune Router-1.5B + AWQ quantization |
| P3 | Dual-GPU vLLM deployment |
| P4 | Gateway server + cascade fallback |
| P5 | Stress test + billing report |
| P6 | Delivery: docs, demo, interview prep |