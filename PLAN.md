# PCA — personal coding assistant

A terminal coding assistant that runs entirely against a **locally-served
LLM**. No cloud calls, no API keys, no telemetry: the model, the prompts, and
the code under discussion never leave the machine.

## Design principles

1. **Local-first.** The backend is any OpenAI-compatible local server
   (Ollama `http://localhost:11434/v1`, LM Studio, llama.cpp's
   `llama-server`, vLLM). PCA speaks the standard `/chat/completions` +
   `/models` endpoints over plain httpx — no vendor SDK, so switching
   runtimes is a base-URL flag.
2. **stdout is the answer.** Only the model's answer is written to stdout;
   all progress, warnings, and errors go to stderr. This makes PCA
   composable: `git commit -F <(pca commit-msg)` works.
3. **Deterministic plumbing, honest failures.** Context gathering (files,
   line ranges, staged diffs) is plain Python. When the backend is down or
   the model isn't pulled, PCA says exactly that (exit 3) with the command
   to fix it — it never half-answers.
4. **Tests never need the model.** Everything is testable with a mock HTTP
   transport; CI runs with no server.

## M1 (this milestone)

- `pca ask "question" [--file PATH ...]` — coding Q&A, optionally grounded
  in files.
- `pca explain PATH [--lines A-B]` — explain a file or a line range.
- `pca review [PATH ...] [--staged]` — review files, or the staged git diff.
- `pca commit-msg` — draft a commit message from the staged diff
  (imperative subject ≤72 chars, body only when the why is non-obvious).
- `pca doctor` — check the backend: server reachable, models available,
  configured model present; exact remediation hints when not.
- Streaming by default (`--no-stream` to disable), `--model`, `--base-url`,
  `--timeout` flags with `PCA_MODEL` / `PCA_BASE_URL` env fallbacks.
- Exit codes: 0 ok · 2 usage error · 3 backend unreachable or model missing
  · 4 nothing to do (e.g. `--staged` with an empty index) · 130 interrupted.
- Doc-contract tests: every CLI flag, command, and exit code documented in
  README is pinned to the real parser.

## M2 candidates (not committed)

- Interactive REPL mode with conversation memory (`pca chat`).
- Repo-aware context: ripgrep/ctags-backed retrieval so `ask` can pull in
  relevant code automatically instead of explicit `--file`.
- `pca review --branch` — review a whole branch diff vs main.
- Config file (`~/.config/pca/config.toml`) for per-machine defaults;
  per-command model overrides (small model for commit-msg, big for review).
- Claude Code skill wrapper (same pattern as maxim's SKILL.md, pinned by
  contract tests).
- Response caching for repeated explains of unchanged files.

## Deliberate non-goals

- No agentic file editing in M1 — PCA advises; the human edits. Multi-file
  edit application is a possible M3, gated on a real need.
- No cloud fallback. If the local server is down, PCA fails honestly rather
  than silently phoning home.
