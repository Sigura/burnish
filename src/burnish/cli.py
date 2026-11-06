import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from burnish import client, desktop
from burnish.config import load_config, setup_logging


def _report_error(message: str) -> None:
    # Vim runs filters with 2>&1, so a message on stderr would land in the buffer.
    merged = not sys.stderr.isatty() and _same_file(sys.stdout, sys.stderr)
    if not merged:
        print(f"burnish: {message}", file=sys.stderr)
        return
    setup_logging("cli")
    logger.error(message)
    try:
        desktop.backend().notify(message, critical=True)
    except (OSError, subprocess.SubprocessError, desktop.DesktopError) as e:
        logger.error("notification failed: {}", e)


def _same_file(a, b) -> bool:
    try:
        first, second = os.fstat(a.fileno()), os.fstat(b.fileno())
    except (OSError, ValueError):
        return False
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="burnish", description="Refine English text from stdin.")
    parser.add_argument("file", nargs="?", type=Path, help="edit this file in place (for use as $EDITOR)")
    parser.add_argument("--mode", choices=["fix", "polish", "all"], default="all")
    parser.add_argument("--diff", action="store_true", help="print a unified diff instead of the text")
    parser.add_argument("--model", help="use this model instead of the automatic choice")
    args = parser.parse_args(argv)

    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    try:
        result = client.refine(load_config().server.url, text, args.mode, args.model)
    except client.RefineError as e:
        _report_error(str(e))
        # As a filter the output replaces the selection, so a failure must give the input back.
        if not args.file:
            sys.stdout.write(text)
        return 1

    if args.diff:
        sys.stdout.write(result["diff"])
    elif args.file:
        args.file.write_text(result["text"], encoding="utf-8")
    else:
        sys.stdout.write(result["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
