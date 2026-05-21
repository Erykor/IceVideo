"""Load config.toml — falls back to packaged defaults."""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_NAME = "config.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _packaged_defaults() -> dict[str, Any]:
    """Read the config.toml that ships with the package (one dir up from this file)."""
    here = Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_NAME
    if not here.exists():
        return {}
    return tomllib.loads(here.read_text())


def load_config(explicit_path: str | None = None, cwd: str | None = None) -> dict[str, Any]:
    """Load config: packaged defaults <- ./config.toml <- explicit_path."""
    cfg = _packaged_defaults()
    if cwd is None:
        cwd = os.getcwd()
    local = Path(cwd) / DEFAULT_CONFIG_NAME
    if local.exists() and str(local) != str(Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_NAME):
        cfg = _deep_merge(cfg, tomllib.loads(local.read_text()))
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        cfg = _deep_merge(cfg, tomllib.loads(p.read_text()))
    return cfg


@dataclass
class Paths:
    # `input_dir` is the *first* input directory (kept for backward compat).
    # `input_dirs` is the canonical list — all are searched for input videos.
    input_dir: Path
    work_dir: Path
    output_dir: Path
    input_glob: str
    input_dirs: list[Path] | None = None

    @classmethod
    def from_config(cls, cfg: dict, base: Path | None = None) -> "Paths":
        base = base or Path.cwd()
        p = cfg["paths"]
        def resolve(s: str) -> Path:
            q = Path(s)
            return (q if q.is_absolute() else (base / q)).resolve()

        # `input_dir` may be a string or a list of strings.
        raw = p.get("input_dirs") or p.get("input_dir") or "."
        if isinstance(raw, str):
            dirs = [resolve(raw)]
        else:
            dirs = [resolve(s) for s in raw]

        return cls(
            input_dir=dirs[0],
            input_dirs=dirs,
            work_dir=resolve(p["work_dir"]),
            output_dir=resolve(p["output_dir"]),
            input_glob=p["input_glob"],
        )

    def subdir(self, name: str) -> Path:
        d = self.work_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def discover_videos(paths: Paths) -> list[Path]:
    """Return sorted list of input videos across all input_dirs.
    `input_glob` may contain multiple space-separated patterns."""
    patterns = paths.input_glob.split()
    dirs = paths.input_dirs or [paths.input_dir]
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        for pat in patterns:
            for p in sorted(d.glob(pat)):
                if p.is_file() and p not in seen:
                    seen.add(p); out.append(p)
    # Sort by filename so multi-dir results have a deterministic chronological order.
    out.sort(key=lambda p: p.name)
    return out
