"""Context gathering is deterministic plumbing — test it like plumbing."""

import subprocess
from pathlib import Path

import pytest

from pca.context import (
    MAX_FILE_BYTES,
    ContextError,
    parse_line_range,
    read_file_block,
    read_files_block,
    staged_diff,
)


def test_file_block_is_fenced_with_language(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")
    block = read_file_block(f)
    assert f"### {f}" in block
    assert "```python" in block
    assert "print('hi')" in block


def test_line_range_selects_inclusive_1_based(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("one\ntwo\nthree\nfour\n")
    block = read_file_block(f, lines=(2, 3))
    assert "two\nthree" in block
    assert "one" not in block.split("```")[1]
    assert "(lines 2-3)" in block


def test_missing_file_raises_context_error(tmp_path: Path):
    with pytest.raises(ContextError, match="no such file"):
        read_file_block(tmp_path / "ghost.py")


def test_bad_line_ranges_are_rejected(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("one\n")
    with pytest.raises(ContextError, match="bad line range"):
        read_file_block(f, lines=(3, 2))
    with pytest.raises(ContextError, match="only 1 lines"):
        read_file_block(f, lines=(5, 9))


def test_oversized_file_is_truncated_and_disclosed(tmp_path: Path):
    f = tmp_path / "big.txt"
    f.write_text("x" * (MAX_FILE_BYTES + 100))
    block = read_file_block(f)
    assert "(truncated" in block
    assert len(block) < MAX_FILE_BYTES + 500


def test_multiple_files_join(tmp_path: Path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("A")
    b.write_text("B")
    block = read_files_block([a, b])
    assert f"### {a}" in block and f"### {b}" in block


def test_parse_line_range_forms():
    assert parse_line_range("10-42") == (10, 42)
    assert parse_line_range("17") == (17, 17)
    with pytest.raises(ContextError, match="bad --lines"):
        parse_line_range("abc")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(cwd),  # ignore the user's global git config
        },
    )


def test_staged_diff_returns_the_staged_change(tmp_path: Path):
    _git(tmp_path, "init", "-b", "main")
    f = tmp_path / "f.txt"
    f.write_text("old\n")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-m", "init")
    f.write_text("new\n")
    _git(tmp_path, "add", "f.txt")
    diff = staged_diff(cwd=tmp_path)
    assert "-old" in diff and "+new" in diff


def test_empty_index_raises_nothing_staged(tmp_path: Path):
    _git(tmp_path, "init", "-b", "main")
    with pytest.raises(ContextError, match="nothing staged"):
        staged_diff(cwd=tmp_path)


def test_not_a_repo_raises_context_error(tmp_path: Path):
    with pytest.raises(ContextError, match="git diff failed"):
        staged_diff(cwd=tmp_path)
