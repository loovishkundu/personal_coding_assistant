"""The LLM client: streaming assembly, error mapping, model discovery."""

import httpx
import pytest
from conftest import FakeServer

from pca.llm import LLM, BackendError, _iter_sse_tokens

BASE = "http://localhost:11434/v1"


def make_llm(server: FakeServer, model: str = "qwen2.5-coder:7b") -> LLM:
    return LLM(BASE, model, timeout_s=5.0, transport=server.transport)


def test_streaming_assembles_and_reports_tokens():
    server = FakeServer(tokens=["def ", "f():", " pass"])
    seen: list[str] = []
    reply = make_llm(server).chat("sys", "user", stream=True, on_token=seen.append)
    assert reply == "def f(): pass"
    assert seen == ["def ", "f():", " pass"]  # every token surfaced as it arrived


def test_non_streaming_returns_full_reply():
    server = FakeServer(tokens=["full", " reply"])
    assert make_llm(server).chat("sys", "user", stream=False) == "full reply"


def test_request_carries_model_and_messages():
    server = FakeServer()
    make_llm(server, model="my-model").chat("SYSTEM", "USER", stream=False)
    import json

    body = json.loads(server.requests[0].content)
    assert body["model"] == "my-model"
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][1] == {"role": "user", "content": "USER"}


def test_connection_refused_becomes_backend_error_with_hint():
    server = FakeServer(raise_transport=True)
    with pytest.raises(BackendError) as exc:
        make_llm(server).chat("s", "u", stream=False)
    assert "cannot reach" in str(exc.value)
    assert exc.value.hint and "pca doctor" in exc.value.hint


def test_missing_model_404_names_the_model_and_the_fix():
    server = FakeServer(
        fail_with=httpx.Response(
            404, json={"error": {"message": 'model "nope" not found, try pulling it first'}}
        )
    )
    with pytest.raises(BackendError) as exc:
        make_llm(server, model="nope").chat("s", "u", stream=False)
    assert "nope" in str(exc.value)
    assert "pull" in (exc.value.hint or "")


def test_500_mentioning_model_is_not_misclassified_as_missing():
    # A real Ollama failure mode: OOM while loading an INSTALLED model.
    # Telling the user to `ollama pull` it would be wrong — this must stay
    # a generic HTTP 500. (Regression: operator precedence in _http_error.)
    server = FakeServer(
        fail_with=httpx.Response(
            500, json={"error": {"message": "failed to load model into memory"}}
        )
    )
    with pytest.raises(BackendError, match="HTTP 500") as exc:
        make_llm(server).chat("s", "u", stream=False)
    assert exc.value.hint is None


def test_read_timeout_is_reported_as_timeout_not_unreachable():
    class TimeoutServer(FakeServer):
        def handler(self, request):
            raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(BackendError, match="timed out") as exc:
        make_llm(TimeoutServer()).chat("s", "u", stream=False)
    assert "--timeout" in (exc.value.hint or "")


def test_client_ignores_proxy_env():
    # PCA only ever talks to localhost; HTTP_PROXY must never route prompts
    # through an external proxy.
    server = FakeServer()
    assert make_llm(server)._client.trust_env is False


def test_streaming_http_error_is_mapped_not_swallowed():
    server = FakeServer(fail_with=httpx.Response(500, json={"error": {"message": "boom"}}))
    with pytest.raises(BackendError, match="HTTP 500"):
        make_llm(server).chat("s", "u", stream=True)


def test_list_models_returns_ids():
    server = FakeServer(models=["a:7b", "b:latest"])
    assert make_llm(server).list_models() == ["a:7b", "b:latest"]


def test_list_models_maps_connection_errors():
    server = FakeServer(raise_transport=True)
    with pytest.raises(BackendError, match="cannot reach"):
        make_llm(server).list_models()


def test_sse_parser_skips_noise_and_stops_at_done():
    lines = iter(
        [
            ": keep-alive comment",
            "",
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"tok"}}]}',
            "data: not-json",
            'data: {"unexpected": true}',
            "data: 42",  # valid JSON, wrong shape: TypeError, must be skipped
            'data: {"choices":[{"delta":null}]}',  # null delta: AttributeError
            'data: {"choices":[]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"after done"}}]}',
        ]
    )
    assert list(_iter_sse_tokens(lines)) == ["tok"]


def test_unexpected_response_shape_is_a_backend_error():
    class WeirdServer(FakeServer):
        def handler(self, request):
            return httpx.Response(200, json={"nope": []})

    with pytest.raises(BackendError, match="unexpected response shape"):
        make_llm(WeirdServer()).chat("s", "u", stream=False)
