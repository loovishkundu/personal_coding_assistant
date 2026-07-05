"""Deterministic context gathering: files, line ranges, staged git diffs.

Everything here is plain Python — what the model sees is exactly what these
functions return, so context bugs are testable without any LLM.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

# Guard against feeding a local model more than it can attend to. Truncation
# is disclosed twice: inline in the prompt block (so the model knows) and via
# the `warn` callback (so the user knows).
MAX_FILE_BYTES = 200_000
MAX_DIFF_BYTES = 200_000

WarnFn = Callable[[str], None]

_LANG_BY_SUFFIX = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}


class ContextError(Exception):
    """A context source is missing or unusable (bad path, empty index, ...)."""


def read_file_block(
    path: Path, lines: tuple[int, int] | None = None, warn: WarnFn | None = None
) -> str:
    """Return one file (or a 1-indexed inclusive line range) as a fenced block."""
    if not path.is_file():
        raise ContextError(f"no such file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ContextError(f"cannot read {path}: {exc}") from exc
    if b"\x00" in data[:8192]:
        raise ContextError(f"{path} looks like a binary file — pca reads source, not binaries")
    text = data.decode("utf-8", errors="replace")

    label = str(path)
    if lines is not None:
        start, end = lines
        all_lines = text.splitlines()
        if start < 1 or end < start:
            raise ContextError(f"bad line range {start}-{end} (must be 1-based, start <= end)")
        if start > len(all_lines):
            raise ContextError(f"{path} has only {len(all_lines)} lines; range starts at {start}")
        text = "\n".join(all_lines[start - 1 : end])
        label = f"{path} (lines {start}-{min(end, len(all_lines))})"

    truncated = False
    if len(text.encode()) > MAX_FILE_BYTES:
        text = text.encode()[:MAX_FILE_BYTES].decode(errors="replace")
        truncated = True

    lang = _LANG_BY_SUFFIX.get(path.suffix.lower(), "")
    block = f"### {label}\n```{lang}\n{text}\n```"
    if truncated:
        block += f"\n(truncated at {MAX_FILE_BYTES // 1000}KB)"
        if warn is not None:
            warn(
                f"{path} exceeds {MAX_FILE_BYTES // 1000}KB — the model sees only the "
                "beginning; pass --lines to pick the part that matters"
            )
    return block


def read_files_block(paths: list[Path], warn: WarnFn | None = None) -> str:
    return "\n\n".join(read_file_block(p, warn=warn) for p in paths)


def parse_line_range(raw: str) -> tuple[int, int]:
    """Parse '10-42' (or a single '17') into a 1-indexed inclusive range."""
    try:
        if "-" in raw:
            start_s, end_s = raw.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(raw)
    except ValueError as exc:
        raise ContextError(f"bad --lines value {raw!r} (expected e.g. 10-42)") from exc
    return start, end


def staged_diff(cwd: Path | None = None, warn: WarnFn | None = None) -> str:
    """Return the staged diff, or raise ContextError when there is nothing to do.

    --no-ext-diff keeps the output a plain unified diff even when the user
    has diff.external (difftastic etc.) configured — external tools emit
    human-oriented, often colored output the model would misread.
    """
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--no-color", "--no-ext-diff"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
        )
    except OSError as exc:  # git not installed
        raise ContextError(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        raise ContextError(f"git diff failed: {proc.stderr.strip() or 'not a git repository?'}")
    diff = proc.stdout
    if not diff.strip():
        raise ContextError("nothing staged — `git add` your changes first")
    if len(diff.encode()) > MAX_DIFF_BYTES:
        diff = diff.encode()[:MAX_DIFF_BYTES].decode(errors="replace")
        diff += f"\n(diff truncated at {MAX_DIFF_BYTES // 1000}KB)"
        if warn is not None:
            warn(
                f"staged diff exceeds {MAX_DIFF_BYTES // 1000}KB — the model sees only "
                "the beginning; consider staging less at once"
            )
    return diff
