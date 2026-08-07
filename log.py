"""Colored, ISO-timestamped logging to stderr.

Levels: INFO (default color), WARNING (yellow), ERROR (red). Emits one line
per record: `<ISO timestamp> <LEVEL> <message>`.
"""

import sys
from datetime import datetime

RESET = "\033[0m"
_COLORS = {
    "INFO": "",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[31m",
}

_TTY = sys.stderr.isatty()


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _emit(level: str, message: str) -> None:
    color = _COLORS[level]
    line = f"{_iso_now()} {level} {message}"
    if _TTY and color:
        line = f"{color}{line}{RESET}"
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def info(message: str) -> None:
    _emit("INFO", message)


def warning(message: str) -> None:
    _emit("WARNING", message)


def error(message: str) -> None:
    _emit("ERROR", message)
