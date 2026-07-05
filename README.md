# PCA — personal coding assistant

A terminal coding assistant that runs entirely against a **locally-served
LLM**. No cloud calls, no API keys, no telemetry — the model, the prompts,
and the code under discussion never leave your machine.

```
┌─────────────┐     OpenAI-compatible API      ┌──────────────────────┐
│  pca (CLI)  │ ──── /chat/completions ──────▶ │  local LLM server    │
│             │ ◀─── streamed tokens ───────── │  Ollama · LM Studio  │
└─────────────┘                                │  llama.cpp · vLLM    │
      │                                        └──────────────────────┘
      └─ deterministic context: files, line ranges, staged git diffs
```

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Any OpenAI-compatible local LLM server. The default configuration targets
  [Ollama](https://ollama.com):

  ```bash
  brew install ollama          # or the installer from ollama.com
  ollama serve                 # if not already running as a service
  ollama pull qwen2.5-coder:7b # the default model; any coding model works
  ```

  LM Studio, llama.cpp (`llama-server`), and vLLM work too — point
  `--base-url` (or `PCA_BASE_URL`) at them.

## Setup

```bash
git clone https://github.com/loovishkundu/pca.git ~/dev/pca
cd ~/dev/pca
uv sync
uv run pca doctor   # verifies the server is reachable and the model is pulled
```

> Keep the checkout on a non-iCloud-synced path (like `~/dev`). iCloud sets
> hidden flags inside `.venv` that silently break editable installs.

## Usage

```bash
# Ask a coding question, optionally grounded in files (--file is repeatable)
uv run pca ask "why would asyncio.gather swallow my exception?"
uv run pca ask "what does this config control?" --file src/pca/config.py

# Explain a file, or just a line range (1-based, inclusive)
uv run pca explain src/pca/llm.py
uv run pca explain src/pca/llm.py --lines 90-140

# Review files, or whatever is staged
uv run pca review src/pca/cli.py
uv run pca review --staged

# Draft a commit message from the staged diff (stdout carries only the
# message, so it pipes straight into git)
git commit -F <(uv run pca commit-msg)

# Check the backend: server reachable, models available, configured model present
uv run pca doctor
```

### Options (all commands)

| Flag | Meaning | Default |
| --- | --- | --- |
| `--model` | model to use | `$PCA_MODEL`, else `qwen2.5-coder:7b` |
| `--base-url` | OpenAI-compatible server URL | `$PCA_BASE_URL`, else `http://localhost:11434/v1` |
| `--timeout` | read timeout in seconds | `300` (cold model loads are slow) |
| `--no-stream` | print the reply only when complete | streaming on |

`ask` also takes `--file PATH` (repeatable); `explain` takes `--lines A-B`;
`review` takes file paths and/or `--staged`.

### Output contract

stdout carries **only the answer** (or `doctor`'s report); all progress and
errors go to stderr. When stdout is a pipe rather than a terminal, PCA
buffers the reply and writes it only after it completed successfully — a
mid-stream backend failure leaves the pipe **empty** (exit 3), never a
truncated message. That is what makes `git commit -F <(pca commit-msg)`
safe: git either gets the whole message or nothing.

### Exit codes

- `0` — success
- `2` — usage error (bad flags/arguments)
- `3` — backend problem: server unreachable, or the model isn't available
  (`pca doctor` tells you which, and how to fix it)
- `4` — input problem: missing file, bad `--lines` range, or nothing staged
- `130` — interrupted (Ctrl-C)

## Development

```bash
uv run isort . && uv run black . && uv run ruff check .   # lint gate
uv run pytest                                             # no server needed
```

Tests run against a mock HTTP transport and a fake LLM — no local model, no
network. `tests/test_docs_contract.py` pins this README's flags, commands,
defaults, and exit codes to the real CLI, so documentation drift fails CI.

## Layout

```
src/pca/
  cli.py       argument parsing, dispatch, exit codes, stdout purity
  llm.py       OpenAI-compatible client (httpx): streaming, error mapping
  context.py   deterministic context: files, line ranges, staged diffs
  prompts.py   one short system prompt per command
  config.py    flags → env → defaults resolution
tests/         mock-transport + fake-LLM suite, incl. doc-contract tests
PLAN.md        design principles, M1 scope, M2 candidates
```
