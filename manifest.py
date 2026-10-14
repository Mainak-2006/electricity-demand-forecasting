"""Per-run manifests: provenance for every pipeline run.

Each run (``data``/``train``/``evaluate``/``predict``) writes a small JSON
manifest capturing what was run, on which code revision, over which inputs, and
how long it took — enough to reproduce or audit a result later:

    {
      "manifest_version": 1,
      "command": "train",
      "status": "ok",
      "started_at": "2026-10-14T13:55:01Z",
      "duration_s": 41.2,
      "git": {"sha": "...", "dirty": false},
      "runtime": {"python": "3.14.8", "platform": "Linux-...", "cwd": "..."},
      "args": {...},
      "inputs": {"data": {"path": "...", "sha256": "...", "bytes": 12345}},
      "outputs": {"model": "artifacts/best_model.pt"},
      "durations": {"train": 39.8},
      "validation": {...}
    }

Manifests land in ``outputs/manifests/<command>_<UTC stamp>.json`` by default.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
DEFAULT_MANIFEST_DIR = "outputs/manifests"
_HASH_CHUNK = 1 << 20  # 1 MiB


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_revision(cwd: str | Path | None = None) -> dict[str, Any] | None:
    """Return ``{"sha": ..., "dirty": bool}`` for the current repo, or None."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if sha.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return {"sha": sha.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, ValueError):  # git missing or not a repo
        return None


def file_sha256(path: str | Path, chunk: int = _HASH_CHUNK) -> str | None:
    """SHA-256 of a file's contents, or None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def file_info(path: str | Path) -> dict[str, Any]:
    """Path, size and content hash for a manifest input."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "bytes": p.stat().st_size,
        "sha256": file_sha256(p),
    }


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of arbitrary values to JSON-serializable data."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    item = getattr(obj, "item", None)  # numpy / torch scalars
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


@dataclass
class RunManifest:
    """Accumulate provenance for one run and write it as JSON."""

    command: str
    args: dict[str, Any] | None = None
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR
    out_path: str | Path | None = None
    enabled: bool = True

    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    started_at: str = field(default_factory=_utc_iso)
    _start: float = field(default_factory=time.perf_counter)

    # --- recording --------------------------------------------------------
    def add_input(self, name: str, path: str | Path | None) -> None:
        if path is not None:
            self.inputs[name] = file_info(path)

    def add_output(self, name: str, path: str | Path | None) -> None:
        if path is not None:
            self.outputs[name] = str(path)

    def set(self, key: str, value: Any) -> None:
        self.extra[key] = value

    def record_duration(self, name: str, seconds: float) -> None:
        self.durations[name] = round(float(seconds), 3)

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    # --- building / writing ----------------------------------------------
    def build(
        self,
        *,
        status: str = "ok",
        error: str | None = None,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "manifest_version": MANIFEST_VERSION,
            "command": self.command,
            "status": status,
            "started_at": self.started_at,
            "duration_s": round(self.elapsed() if duration_s is None else duration_s, 3),
            "git": git_revision(),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cwd": str(Path.cwd()),
            },
            "args": to_jsonable(self.args or {}),
            "inputs": to_jsonable(self.inputs),
            "outputs": to_jsonable(self.outputs),
            "durations": to_jsonable(self.durations),
        }
        if error:
            record["error"] = error
        if self.extra:
            record["extra"] = to_jsonable(self.extra)
        return record

    def path(self) -> Path:
        if self.out_path is not None:
            return Path(self.out_path)
        return Path(self.manifest_dir) / f"{self.command}_{_utc_stamp()}.json"

    def write(
        self,
        *,
        status: str = "ok",
        error: str | None = None,
        duration_s: float | None = None,
    ) -> Path | None:
        """Write the manifest. A no-op when ``enabled`` is False."""
        if not self.enabled:
            return None
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = self.build(status=status, error=error, duration_s=duration_s)
        path.write_text(json.dumps(record, indent=2))
        return path

    # --- context-manager sugar -------------------------------------------
    def __enter__(self) -> RunManifest:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        error = None if exc is None else f"{type(exc).__name__}: {exc}"
        try:
            self.write(status="ok" if exc is None else "error", error=error)
        except OSError:  # never mask the real failure with a manifest error
            pass
        return False


def add_manifest_arguments(
    parser,
    default_dir: str | Path = DEFAULT_MANIFEST_DIR,
) -> None:
    """Attach the shared ``--manifest-dir``/``--no-manifest`` options."""
    parser.add_argument(
        "--manifest-dir",
        default=str(default_dir),
        help="directory for per-run manifest JSON files",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write a run manifest",
    )
