"""Bounded storage helpers for the public Studio demo."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable


def positive_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer setting without making startup fragile."""
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


class OutputJanitor:
    """Remove expired outputs and enforce count and size limits."""

    def __init__(
        self,
        directory: str | Path,
        *,
        retention_seconds: int,
        max_bytes: int,
        max_files: int,
        check_interval_seconds: int,
    ) -> None:
        self.directory = Path(directory)
        self.retention_seconds = retention_seconds
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.check_interval_seconds = check_interval_seconds
        self._last_check = 0.0
        self._lock = threading.Lock()

    def cleanup(
        self,
        *,
        force: bool = False,
        keep: Iterable[str | Path] = (),
    ) -> tuple[int, int]:
        """Delete stale/old WAVs and return ``(file_count, total_bytes)``."""
        now = time.time()
        if not force and now - self._last_check < self.check_interval_seconds:
            return 0, 0

        with self._lock:
            now = time.time()
            if not force and now - self._last_check < self.check_interval_seconds:
                return 0, 0
            self._last_check = now
            self.directory.mkdir(parents=True, exist_ok=True)
            protected = {Path(path).resolve() for path in keep}

            entries: list[tuple[Path, float, int]] = []
            for path in self.directory.glob("spk_*.wav"):
                try:
                    stat = path.stat()
                    resolved = path.resolve()
                    if resolved not in protected and now - stat.st_mtime > self.retention_seconds:
                        path.unlink(missing_ok=True)
                        continue
                    entries.append((path, stat.st_mtime, stat.st_size))
                except OSError as exc:
                    print(f">> output cleanup skipped {path}: {exc}")

            total_bytes = sum(size for _, _, size in entries)
            file_count = len(entries)
            for path, _, size in sorted(entries, key=lambda item: item[1]):
                if file_count <= self.max_files and total_bytes <= self.max_bytes:
                    break
                if path.resolve() in protected:
                    continue
                try:
                    path.unlink(missing_ok=True)
                    file_count -= 1
                    total_bytes -= size
                except OSError as exc:
                    print(f">> output cleanup skipped {path}: {exc}")

            return file_count, total_bytes
