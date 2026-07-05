"""CLI: argument parsing, dispatch, exit codes.

Contract for wrappers and pipes:
- stdout carries the answer only (the model's reply, or doctor's report);
  all progress and errors go to stderr — `git commit -F <(pca commit-msg)`
  must receive nothing but the message
- exit codes: 0 ok · 2 usage error (argparse) · 3 backend unreachable or
  model missing · 4 input problem (missing file, empty staged index, bad
  line range) · 130 interrupted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, Settings
from .context import (
    ContextError,
    parse_line_range,
    read_file_block,
    read_files_block,
    staged_diff,
)
from .llm import LLM, BackendError
from .prompts import ASK_SYSTEM, COMMIT_MSG_SYSTEM, EXPLAIN_SYSTEM, REVIEW_SYSTEM

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BACKEND = 3
EXIT_NO_INPUT = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pca",
        description=(
            "Personal coding assistant backed by a locally-served LLM. "
            "Works with any OpenAI-compatible server (Ollama, LM Studio, "
            "llama.cpp, vLLM); nothing leaves your machine."
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        help=f"model to use (default: $PCA_MODEL or {DEFAULT_MODEL})",
    )
    common.add_argument(
        "--base-url",
        help=f"OpenAI-compatible server URL (default: $PCA_BASE_URL or {DEFAULT_BASE_URL})",
    )
    common.add_argument(
        "--timeout",
        type=float,
        help="read timeout in seconds (default: 300; cold model loads are slow)",
    )
    common.add_argument(
        "--no-stream",
        action="store_true",
        help="print the reply once it is complete instead of streaming tokens",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", parents=[common], help="ask a coding question")
    ask.add_argument("question", help="the question")
    ask.add_argument(
        "--file",
        dest="files",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="ground the answer in this file (repeatable)",
    )

    explain = sub.add_parser("explain", parents=[common], help="explain a file or line range")
    explain.add_argument("path", type=Path, help="file to explain")
    explain.add_argument("--lines", metavar="A-B", help="1-based inclusive line range, e.g. 10-42")

    review = sub.add_parser("review", parents=[common], help="review files or the staged diff")
    review.add_argument("paths", nargs="*", type=Path, help="files to review")
    review.add_argument("--staged", action="store_true", help="review the staged git diff")

    sub.add_parser(
        "commit-msg",
        parents=[common],
        help="draft a commit message from the staged diff",
    )

    sub.add_parser(
        "doctor",
        parents=[common],
        help="check the backend: server reachable, models available, configured model present",
    )
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        stream=not args.no_stream,
    )


def _stdout_is_tty() -> bool:
    return sys.stdout.isatty()


def _run_chat(settings: Settings, system: str, user: str, console: Console) -> int:
    console.print(f"[dim]pca · {escape(settings.model)} @ {escape(settings.base_url)}[/dim]")
    llm = LLM(settings.base_url, settings.model, settings.timeout_s)
    # Stream tokens live only when a human is watching. When stdout is a pipe
    # (`git commit -F <(pca commit-msg)`), buffer and write only after the
    # reply completed: a mid-stream failure must leave the pipe EMPTY — the
    # pipe consumer cannot see pca's exit code, so a truncated message would
    # get used as if it were the real thing.
    live = settings.stream and _stdout_is_tty()
    try:
        if live:
            reply = llm.chat(
                system, user, stream=True, on_token=lambda t: print(t, end="", flush=True)
            )
            if reply and not reply.endswith("\n"):
                print()
        else:
            reply = llm.chat(system, user, stream=settings.stream)
            if reply.strip():
                print(reply.rstrip("\n"))
    finally:
        llm.close()
    if not reply.strip():
        console.print("[yellow]warning:[/yellow] the model returned an empty reply")
    return EXIT_OK


def _warn_fn(console: Console):
    return lambda message: console.print(f"[yellow]warning:[/yellow] {escape(message)}")


def _cmd_ask(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    user = args.question
    if args.files:
        user += "\n\nRelevant files:\n\n" + read_files_block(args.files, warn=_warn_fn(console))
    return _run_chat(settings, ASK_SYSTEM, user, console)


def _cmd_explain(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    lines = parse_line_range(args.lines) if args.lines else None
    user = "Explain this code.\n\n" + read_file_block(args.path, lines, warn=_warn_fn(console))
    return _run_chat(settings, EXPLAIN_SYSTEM, user, console)


def _cmd_review(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    warn = _warn_fn(console)
    sections: list[str] = []
    if args.staged:
        sections.append("### staged diff\n```diff\n" + staged_diff(warn=warn) + "\n```")
    if args.paths:
        sections.append(read_files_block(args.paths, warn=warn))
    user = "Review the following.\n\n" + "\n\n".join(sections)
    return _run_chat(settings, REVIEW_SYSTEM, user, console)


def _cmd_commit_msg(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    user = "Staged diff:\n\n```diff\n" + staged_diff(warn=_warn_fn(console)) + "\n```"
    return _run_chat(settings, COMMIT_MSG_SYSTEM, user, console)


def _cmd_doctor(args: argparse.Namespace, settings: Settings, console: Console) -> int:
    llm = LLM(settings.base_url, settings.model, settings.timeout_s)
    try:
        try:
            models = llm.list_models()
        except BackendError as exc:
            print(f"server:  UNREACHABLE — {exc}")
            if exc.hint:
                print(f"hint:    {exc.hint}")
            return EXIT_BACKEND
    finally:
        llm.close()

    print(f"server:  ok — {settings.base_url}")
    if models:
        print(f"models:  {len(models)} available")
        for m in models:
            print(f"  - {m}")
    else:
        print("models:  none installed")
    # Tag tolerance covers exactly Ollama's default: `ollama pull foo`
    # registers "foo:latest", which a bare "foo" resolves to. Other tags do
    # NOT resolve from a bare name, so they must not satisfy the check.
    names = set(models) | {m.removesuffix(":latest") for m in models if m.endswith(":latest")}
    if settings.model in names:
        print(f"model:   '{settings.model}' is available — you're good")
        return EXIT_OK
    print(f"model:   '{settings.model}' is NOT available")
    print(f"hint:    pull it (e.g. `ollama pull {settings.model}`), or pass --model / set")
    print("         PCA_MODEL to one of the models listed above")
    return EXIT_BACKEND


_COMMANDS = {
    "ask": _cmd_ask,
    "explain": _cmd_explain,
    "review": _cmd_review,
    "commit-msg": _cmd_commit_msg,
    "doctor": _cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "review" and not (args.staged or args.paths):
        parser.error("review needs file paths or --staged")

    settings = _settings(args)
    # Errors must always reach stderr; stdout stays reserved for the answer.
    console = Console(stderr=True)
    try:
        return _COMMANDS[args.command](args, settings, console)
    except ContextError as exc:
        # escape(): error text carries untrusted content (git stderr, paths)
        # that Rich would otherwise parse as markup — and crash on.
        console.print(f"[bold red]error:[/bold red] {escape(str(exc))}")
        return EXIT_NO_INPUT
    except BackendError as exc:
        console.print(f"[bold red]error:[/bold red] {escape(str(exc))}")
        if exc.hint:
            console.print(f"[yellow]hint:[/yellow] {escape(exc.hint)}")
        return EXIT_BACKEND
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
