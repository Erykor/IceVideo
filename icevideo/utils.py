"""Numeric helpers and small ffmpeg/ffprobe wrappers."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sd = float(np.std(x)) or 1.0
    return (x - float(np.mean(x))) / sd


def smooth(x, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < k:
        return x
    kern = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kern, mode="same")


def cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def probe_duration(video: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]).strip()
    return float(out)


def run_ffmpeg(args: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """ffmpeg with sane defaults (stdin closed, quiet)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *args]
    return subprocess.run(cmd, capture_output=capture, text=capture, check=check)


def run_ffmpeg_with_stderr(args: list[str]) -> str:
    """Run ffmpeg at info level and return captured stderr (for parsing filter metadata)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stderr


def video_basename(video: Path) -> str:
    return video.stem


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def save_json(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)


def log(msg: str, prefix: str = "") -> None:
    if prefix:
        print(f"[{prefix}] {msg}", flush=True)
    else:
        print(msg, flush=True)
