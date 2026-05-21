"""Test fixtures — synthesize tiny videos so tests don't need real GoPro footage."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _make_video(out: Path, *, dur: float = 4.0, color: str = "blue", tone_hz: int = 440) -> None:
    """Make a tiny test mp4: color background + sine tone."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d={dur}:r=24",
        "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:duration={dur}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        "-shortest", str(out),
    ], check=True)


@pytest.fixture(scope="session")
def fixtures_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("icevideo_fixtures")
    return root


@pytest.fixture(scope="session")
def sample_video(fixtures_root: Path) -> Path:
    v = fixtures_root / "sample.mp4"
    if not v.exists():
        _make_video(v, dur=4.0, color="red", tone_hz=440)
    return v


@pytest.fixture(scope="session")
def sample_video2(fixtures_root: Path) -> Path:
    v = fixtures_root / "sample2.mp4"
    if not v.exists():
        _make_video(v, dur=3.0, color="green", tone_hz=660)
    return v


@pytest.fixture(scope="session")
def sample_video3(fixtures_root: Path) -> Path:
    v = fixtures_root / "sample3.mp4"
    if not v.exists():
        _make_video(v, dur=2.5, color="blue", tone_hz=880)
    return v


@pytest.fixture(scope="session", autouse=True)
def _check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH", allow_module_level=True)
