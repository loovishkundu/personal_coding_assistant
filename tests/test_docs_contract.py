"""README is a contract, not prose: pin it to the real CLI.

The mechanical half of CLAUDE.md's "verify README before every push" rule —
flags, commands, defaults, and exit codes in the README must match the code,
and the README must not advertise flags that don't exist.
"""

import re
from pathlib import Path

from pca import cli
from pca.config import DEFAULT_BASE_URL, DEFAULT_MODEL

README = (Path(__file__).parent.parent / "README.md").read_text()


def _parser_flags() -> set[str]:
    """Every long option string across the main parser and all subparsers."""
    flags: set[str] = set()
    parser = cli._build_parser()
    parsers = [parser] + [
        sub
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
        for sub in action.choices.values()
    ]
    for p in parsers:
        for action in p._actions:
            flags.update(s for s in action.option_strings if s.startswith("--"))
    flags.discard("--help")
    return flags


def _commands() -> set[str]:
    parser = cli._build_parser()
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            return set(action.choices)
    raise AssertionError("no subparsers found")


def test_every_real_flag_is_documented():
    missing = {f for f in _parser_flags() if f not in README}
    assert not missing, f"README does not document: {sorted(missing)}"


def test_readme_mentions_no_fake_flags():
    documented = set(re.findall(r"(?<![\w-])--[a-z][a-z-]+", README))
    real = _parser_flags()
    fake = documented - real
    assert not fake, f"README documents flags that do not exist: {sorted(fake)}"


def test_every_command_is_documented():
    missing = {c for c in _commands() if f"pca {c}" not in README}
    assert not missing, f"README does not document commands: {sorted(missing)}"


def test_defaults_in_readme_match_config():
    assert DEFAULT_MODEL in README
    assert DEFAULT_BASE_URL in README


def test_exit_codes_documented():
    for code in (cli.EXIT_OK, cli.EXIT_USAGE, cli.EXIT_BACKEND, cli.EXIT_NO_INPUT):
        assert re.search(rf"`{code}`", README), f"exit code {code} not documented in README"
    assert "130" in README


def test_claude_md_lint_gate_matches_reality():
    claude_md = (Path(__file__).parent.parent / "CLAUDE.md").read_text()
    for tool in ("isort", "black", "ruff"):
        assert tool in claude_md
