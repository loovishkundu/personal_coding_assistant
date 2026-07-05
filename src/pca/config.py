"""Runtime configuration: flags win, then environment, then defaults.

PCA talks to any OpenAI-compatible local server. The defaults target LM
Studio's local server (:1234/v1); Ollama (:11434/v1), llama-server
(:8080/v1), and vLLM (:8000/v1) all work by pointing --base-url at them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen3-coder-30b-a3b-instruct-mlx"
# Local models can be slow to first token on cold start (model load into
# RAM/VRAM); the read timeout must cover that, not just steady-state decoding.
DEFAULT_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S
    stream: bool = True

    @classmethod
    def from_env(
        cls,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        stream: bool = True,
    ) -> Settings:
        return cls(
            base_url=(base_url or os.environ.get("PCA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            model=model or os.environ.get("PCA_MODEL") or DEFAULT_MODEL,
            timeout_s=timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S,
            stream=stream,
        )
