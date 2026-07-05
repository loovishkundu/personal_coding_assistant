# PCA — project instructions

## Commits

- Keep commits **small and atomic**: one self-contained change per commit
  (a feature slice, a fix, a doc update — never several unrelated things).
  Prefer a sequence of small commits over one big one.

## Commit messages

- Never include Claude's name in commit messages — no "Co-Authored-By: Claude"
  trailers, no "Generated with Claude" lines, no AI attribution of any kind.
- Be **very precise**: one imperative subject line (≤72 chars) that states
  exactly what changed. Body only when the *why* is non-obvious — and then at
  most a couple of lines. No essays, no restating the diff.

## Linting (mandatory before every commit and before every push)

Always run all three, in this order, and fix anything they report **before**
committing — and run them again before the final push:

```bash
uv run isort .
uv run black .
uv run ruff check .
```

All are configured in `pyproject.toml` (line length 100, py312); isort uses
the black profile so the tools agree with each other.

## Before every push

- Update **README.md** to reflect whatever is being pushed, and verify it
  against the actual code/CLI behavior (commands, flags, exit codes, layout).
  `tests/test_docs_contract.py` pins the mechanical parts; still read it.
- Re-run the lint gate (above) and `uv run pytest` one final time.

## Testing

`uv run pytest` must be green before committing. Tests use a mock HTTP
transport and fake LLM responses — no local model or server needed.

## Environment notes

- Keep this repo on a non-iCloud-synced path (it lives at `~/dev/pca`).
  iCloud-synced locations (Desktop/Documents) set hidden flags inside `.venv`
  that make Python 3.12 silently skip `.pth` files, breaking the editable
  install — this bit maxim before it moved to `~/dev`.
- PCA needs a locally-served LLM with an OpenAI-compatible API (Ollama,
  LM Studio, llama.cpp server, vLLM). No cloud calls, no API keys. Tests
  never touch the server.
