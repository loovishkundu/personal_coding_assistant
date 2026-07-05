"""Shared fakes: an SSE-speaking mock transport and a canned-reply LLM.

No test in this suite talks to a real server or needs a model installed.
"""

from __future__ import annotations

import json

import httpx
import pytest


def sse_stream(tokens: list[str]) -> bytes:
    """Build an OpenAI-style SSE body that streams the given tokens."""
    lines = [
        # Role-only first chunk, as real servers send it.
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
    ]
    for tok in tokens:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": tok}}]}))
    lines.append('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}')
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode()


def chat_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def models_response(ids: list[str]) -> dict:
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


class FakeServer:
    """httpx.MockTransport handler emulating an OpenAI-compatible server."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        models: list[str] | None = None,
        fail_with: httpx.Response | None = None,
        raise_transport: bool = False,
    ):
        self.tokens = tokens if tokens is not None else ["hello", " world"]
        self.models = models if models is not None else ["qwen2.5-coder:7b"]
        self.fail_with = fail_with
        self.raise_transport = raise_transport
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raise_transport:
            raise httpx.ConnectError("connection refused", request=request)
        if self.fail_with is not None:
            return self.fail_with
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=models_response(self.models))
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_stream(self.tokens),
            )
        return httpx.Response(200, json=chat_response("".join(self.tokens)))

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class FakeLLM:
    """Drop-in for cli.LLM: canned reply, records what it was asked."""

    reply = "canned reply"
    models = ["qwen2.5-coder:7b"]
    calls: list[dict] = []

    def __init__(self, base_url, model, timeout_s, transport=None):
        self.base_url = base_url
        self.model = model

    def chat(self, system, user, stream=True, on_token=None):
        FakeLLM.calls.append({"system": system, "user": user, "stream": stream})
        if stream and on_token is not None:
            for chunk in self.reply.split(" "):
                on_token(chunk + " ")
        return self.reply

    def list_models(self):
        return list(FakeLLM.models)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset_fake_llm():
    FakeLLM.calls = []
    FakeLLM.reply = "canned reply"
    FakeLLM.models = ["qwen2.5-coder:7b"]
    yield


@pytest.fixture
def fake_cli_llm(monkeypatch):
    """Route the CLI's LLM class to the fake; returns the class for asserts."""
    import pca.cli as cli

    monkeypatch.setattr(cli, "LLM", FakeLLM)
    return FakeLLM
