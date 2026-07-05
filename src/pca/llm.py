"""Client for any OpenAI-compatible local LLM server, over plain httpx.

Endpoints used: POST {base_url}/chat/completions (SSE when streaming) and
GET {base_url}/models. No SDK: the surface PCA needs is two routes, and
staying on raw httpx keeps every local runtime (Ollama, LM Studio,
llama.cpp, vLLM) equally supported.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import httpx


class BackendError(Exception):
    """The local LLM server is unreachable, or the request failed.

    `hint` carries the remediation the CLI shows the user (start the server,
    pull the model, ...) so error text stays actionable, not generic.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


def _connect_error(base_url: str, exc: Exception) -> BackendError:
    return BackendError(
        f"cannot reach the local LLM server at {base_url} ({exc.__class__.__name__})",
        hint=(
            "start your local server first — e.g. `ollama serve` (Ollama), or point "
            "--base-url / PCA_BASE_URL at LM Studio, llama.cpp --server, or vLLM. "
            "`pca doctor` shows what PCA can see."
        ),
    )


def _http_error(exc: httpx.HTTPStatusError, model: str) -> BackendError:
    status = exc.response.status_code
    detail = ""
    try:
        detail = exc.response.json().get("error", {}).get("message", "")
    except Exception:  # non-JSON error body; the status alone is enough
        detail = exc.response.text[:200]
    if status == 404 and model in detail or "model" in detail.lower():
        return BackendError(
            f"the server rejected model '{model}': {detail or f'HTTP {status}'}",
            hint=f"pull it first (e.g. `ollama pull {model}`) or pass --model / set PCA_MODEL "
            "to a model `pca doctor` lists as available.",
        )
    return BackendError(f"the server returned HTTP {status}: {detail or exc.request.url}")


class LLM:
    """Thin chat client. A custom transport is injectable for tests."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Connect fast-fails when the server is down; read stays generous
        # because a cold model load can take a while before the first token.
        timeout = httpx.Timeout(connect=5.0, read=timeout_s, write=30.0, pool=5.0)
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    # -- chat ---------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Run one exchange and return the full reply.

        With stream=True, `on_token` receives each token as it arrives; the
        assembled reply is still returned so callers never re-join chunks.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if stream:
            return self._chat_stream(messages, on_token)
        return self._chat_once(messages)

    def _chat_once(self, messages: list[dict[str, str]]) -> str:
        try:
            resp = self._client.post(
                "/chat/completions",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc, self.model) from exc
        except httpx.TransportError as exc:
            raise _connect_error(self.base_url, exc) from exc
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise BackendError(f"unexpected response shape from the server: {exc}") from exc

    def _chat_stream(
        self, messages: list[dict[str, str]], on_token: Callable[[str], None] | None
    ) -> str:
        parts: list[str] = []
        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                json={"model": self.model, "messages": messages, "stream": True},
            ) as resp:
                if resp.status_code >= 400:
                    # Read the body before raising so the error detail survives.
                    resp.read()
                    raise _http_error(
                        httpx.HTTPStatusError("", request=resp.request, response=resp),
                        self.model,
                    )
                for token in _iter_sse_tokens(resp.iter_lines()):
                    parts.append(token)
                    if on_token is not None:
                        on_token(token)
        except httpx.TransportError as exc:
            raise _connect_error(self.base_url, exc) from exc
        return "".join(parts)

    # -- discovery ----------------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BackendError(f"the server returned HTTP {exc.response.status_code}") from exc
        except httpx.TransportError as exc:
            raise _connect_error(self.base_url, exc) from exc
        try:
            return [m["id"] for m in resp.json()["data"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"unexpected /models response shape: {exc}") from exc


def _iter_sse_tokens(lines: Iterator[str]) -> Iterator[str]:
    """Yield content tokens from an OpenAI-style SSE stream.

    Lines look like `data: {json}`, the stream ends with `data: [DONE]`.
    Malformed lines are skipped rather than fatal: some servers interleave
    keep-alive comments or empty deltas (role-only first chunk, finish chunk).
    """
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        try:
            delta = json.loads(payload)["choices"][0].get("delta", {})
        except (ValueError, KeyError, IndexError):
            continue
        token = delta.get("content")
        if token:
            yield token
