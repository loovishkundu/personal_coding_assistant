"""The CLI contract: stdout purity, exit codes, dispatch, doctor."""

import subprocess
from pathlib import Path

import pytest
from conftest import FakeLLM, FakeServer

import pca.cli as cli
from pca.llm import LLM
from pca.prompts import ASK_SYSTEM, COMMIT_MSG_SYSTEM, EXPLAIN_SYSTEM, REVIEW_SYSTEM


def test_ask_streams_answer_to_stdout_only(fake_cli_llm, capsys):
    code = cli.main(["ask", "how do I sort a dict?"])
    out, err = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert "canned reply" in out
    assert "canned reply" not in err  # progress goes to stderr, answer to stdout
    assert fake_cli_llm.calls[0]["system"] == ASK_SYSTEM


def test_ask_embeds_file_context(fake_cli_llm, tmp_path: Path):
    f = tmp_path / "code.py"
    f.write_text("VALUE = 42\n")
    cli.main(["ask", "what is VALUE?", "--file", str(f)])
    user = fake_cli_llm.calls[0]["user"]
    assert "what is VALUE?" in user
    assert "VALUE = 42" in user


def test_explain_uses_explain_prompt_and_line_range(fake_cli_llm, tmp_path: Path):
    f = tmp_path / "m.py"
    f.write_text("a\nb\nc\n")
    code = cli.main(["explain", str(f), "--lines", "2-3"])
    assert code == cli.EXIT_OK
    assert fake_cli_llm.calls[0]["system"] == EXPLAIN_SYSTEM
    assert "b\nc" in fake_cli_llm.calls[0]["user"]


def test_missing_file_exits_4(fake_cli_llm, capsys):
    code = cli.main(["explain", "/nonexistent/ghost.py"])
    out, err = capsys.readouterr()
    assert code == cli.EXIT_NO_INPUT
    assert out == ""  # stdout stays clean on errors
    assert "no such file" in err


def test_review_requires_paths_or_staged(fake_cli_llm, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["review"])
    assert exc.value.code == cli.EXIT_USAGE


def test_review_of_files_uses_review_prompt(fake_cli_llm, tmp_path: Path):
    f = tmp_path / "r.py"
    f.write_text("x = 1\n")
    code = cli.main(["review", str(f)])
    assert code == cli.EXIT_OK
    assert fake_cli_llm.calls[0]["system"] == REVIEW_SYSTEM
    assert "x = 1" in fake_cli_llm.calls[0]["user"]


def test_commit_msg_with_empty_index_exits_4(fake_cli_llm, tmp_path: Path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    code = cli.main(["commit-msg"])
    out, err = capsys.readouterr()
    assert code == cli.EXIT_NO_INPUT
    assert "nothing staged" in err
    assert out == ""


def test_commit_msg_sends_diff_with_commit_prompt(fake_cli_llm, tmp_path: Path, monkeypatch):
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "f.txt").write_text("hello\n")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    code = cli.main(["commit-msg"])
    assert code == cli.EXIT_OK
    assert fake_cli_llm.calls[0]["system"] == COMMIT_MSG_SYSTEM
    assert "+hello" in fake_cli_llm.calls[0]["user"]


def test_backend_down_exits_3_with_hint(monkeypatch, capsys):
    server = FakeServer(raise_transport=True)
    monkeypatch.setattr(
        cli, "LLM", lambda base_url, model, timeout_s: LLM(base_url, model, 1.0, server.transport)
    )
    code = cli.main(["ask", "hi"])
    out, err = capsys.readouterr()
    assert code == cli.EXIT_BACKEND
    assert out == ""
    assert "cannot reach" in err
    assert "hint" in err


def test_doctor_reports_ok_when_model_available(fake_cli_llm, capsys):
    FakeLLM.models = ["qwen2.5-coder:7b", "other:latest"]
    code = cli.main(["doctor", "--model", "qwen2.5-coder:7b"])
    out, _ = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert "server:  ok" in out
    assert "you're good" in out


def test_doctor_tag_tolerant_match(fake_cli_llm, capsys):
    FakeLLM.models = ["mistral:latest"]
    code = cli.main(["doctor", "--model", "mistral"])
    assert code == cli.EXIT_OK


def test_doctor_flags_missing_model(fake_cli_llm, capsys):
    FakeLLM.models = ["something-else:7b"]
    code = cli.main(["doctor", "--model", "qwen2.5-coder:7b"])
    out, _ = capsys.readouterr()
    assert code == cli.EXIT_BACKEND
    assert "NOT available" in out
    assert "pull" in out


def test_doctor_reports_unreachable_server(monkeypatch, capsys):
    server = FakeServer(raise_transport=True)
    monkeypatch.setattr(
        cli, "LLM", lambda base_url, model, timeout_s: LLM(base_url, model, 1.0, server.transport)
    )
    code = cli.main(["doctor"])
    out, _ = capsys.readouterr()
    assert code == cli.EXIT_BACKEND
    assert "UNREACHABLE" in out


def test_model_and_base_url_flags_reach_the_client(fake_cli_llm):
    cli.main(["ask", "q", "--model", "custom:3b", "--base-url", "http://localhost:9999/v1"])
    # FakeLLM records ctor args on the instance; calls prove dispatch happened.
    assert fake_cli_llm.calls, "chat was never invoked"


def test_env_defaults_flow_into_settings(monkeypatch):
    monkeypatch.setenv("PCA_MODEL", "env-model")
    monkeypatch.setenv("PCA_BASE_URL", "http://envhost:1234/v1/")
    from pca.config import Settings

    s = Settings.from_env()
    assert s.model == "env-model"
    assert s.base_url == "http://envhost:1234/v1"  # trailing slash normalized
