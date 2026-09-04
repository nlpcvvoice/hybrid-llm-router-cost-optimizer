#!/usr/bin/env python3
"""OpenRouter free-model client with primary + fallback auto-rotation.

Loads the API key from the shared `.env` at runtime (memory only, never
logged/committed). Uses fixed free model IDs (primary -> fallbacks), and
leaves an extensible slot so an enterprise can swap in any paid model later.

SECURITY (mandatory): never print/log the key value; only `key_set=True/False`.
"""
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# The shared .env lives outside the git repo. Path is the known live source.
ENV_PATH = Path("/home/jupyter/opencode/test/.env")

DEFAULT_MODELS = {
    "primary_free": "minimax/minimax-m3:free",
    "fallback_free": [
        "z-ai/glm-5.2:free",
        "google/gemma-4-31b-it:free",
        "nemotron-3-super-120b-a12b:free",
    ],
}


class OpenRouterClient:
    """Thin wrapper around OpenRouter's OpenAI-compatible chat endpoint."""

    def __init__(self, env_path: str | Path = ENV_PATH,
                 models: Optional[dict] = None,
                 timeout: int = 120, max_retries: int = 3):
        load_dotenv(dotenv_path=Path(env_path), override=True)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.base_url = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.timeout = timeout
        self.max_retries = max_retries
        self.models = models or DEFAULT_MODELS
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @property
    def key_set(self) -> bool:
        return bool(self.api_key)

    def _post(self, messages, model, temperature=0.0, json_mode=False,
              max_tokens=1024):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat(self, messages, temperature=0.0, json_mode=False,
             max_tokens=1024, models=None) -> str:
        """Call the primary free model, auto-rotate to fallbacks on failure.

        Resets to the primary on the next call (each call starts fresh).
        `models` optionally overrides {primary_free, fallback_free}.
        """
        errors = []
        m = models or self.models
        attempts = [m["primary_free"]] + m["fallback_free"]
        for model in attempts:
            for attempt in range(1, self.max_retries + 1):
                try:
                    return self._post(messages, model, temperature,
                                      json_mode, max_tokens)
                except (requests.HTTPError, requests.ConnectionError,
                        requests.Timeout, requests.RequestException) as e:
                    errors.append(f"{model}#{attempt}: {e}")
                    status = getattr(e.response, "status_code", None)
                    # 429/5xx rate limits -> backoff, then try next model
                    if status in (402, 429, 502, 503):
                        time.sleep(1.0 + random.random() * 2.0)
                        if status == 429:
                            continue  # retry same model after backoff
                        break  # non-rate-limit error -> move to next model
        raise RuntimeError(
            f"All OpenRouter free models failed. key_set={self.key_set} "
            f"last_errors={errors[-3:]}")

    def list_free_models(self):
        """Return models where pricing.prompt==0 and completion==0."""
        r = self._session.get(f"{self.base_url}/models", timeout=self.timeout)
        r.raise_for_status()
        free = []
        for m in r.json().get("data", []):
            p = m.get("pricing", {})
            if str(p.get("prompt")) == "0" and str(p.get("completion")) == "0":
                free.append(m["id"])
        return free
