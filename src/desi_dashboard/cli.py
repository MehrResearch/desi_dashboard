"""Console entry points for launching the marimo DESI-MS dashboard."""

import sys
from pathlib import Path


def _notebook_path() -> Path:
    return Path(__file__).resolve().parent / "dashboard.py"


def _launch(command: str) -> int:
    """Launch marimo and forward any additional command-line arguments."""
    from marimo._cli.cli import main as marimo_main

    sys.argv = [
        "marimo",
        command,
        "--sandbox",
        str(_notebook_path()),
        *sys.argv[1:],
    ]
    return marimo_main()


def main() -> int:
    """Run the dashboard as a marimo application."""
    return _launch("run")


def edit() -> int:
    """Open the dashboard in the marimo editor."""
    return _launch("edit")


if __name__ == "__main__":
    raise SystemExit(main())
