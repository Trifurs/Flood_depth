"""File/console logging and append-only metric CSV utilities."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

from utils.misc import atomic_write_text


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(row):
                raise ValueError(
                    f"CSV schema changed for {path}: {reader.fieldnames} != {list(row)}"
                )
            existing = list(reader)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerows(existing)
    writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
