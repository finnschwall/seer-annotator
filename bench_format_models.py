#!/usr/bin/env python3
"""Compatibility wrapper for the new SQLite benchmark commands."""

from __future__ import annotations

import sys
import warnings

from seer_annotator.cli import cli


def main() -> None:
    warnings.warn(
        "bench_format_models.py is deprecated; use `seer-annotate benchmark` "
        "(import-db, run, evaluate) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    cli.main(args=["benchmark", *sys.argv[1:]], prog_name="seer-annotate", standalone_mode=True)


if __name__ == "__main__":
    main()
