# Hybrid LLM Router & Cost Optimizer

A smart routing + cost-optimization gateway for LLM APIs: fine-tune a small routing model (Qwen-1.5B) to judge request difficulty at a glance, send simple requests to a local 7B model and complex ones to an OpenRouter free model, with cascading fallback and FinOps billing analytics. Built entirely with free models; the interface is extensible to any paid model later.

## Status

- Project scaffold established (see structure below)
- In development: Phase 1 data engineering

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

## Roadmap

| Phase | Scope |
|-------|-------|
| P1 | Data engineering: seed extraction, labeling, calibration |
| P2 | SFT fine-tune Router-1.5B + AWQ quantization |
| P3 | Dual-GPU vLLM deployment |
| P4 | Gateway server + cascade fallback |
| P5 | Stress test + billing report |
| P6 | Delivery: docs, demo, interview prep |