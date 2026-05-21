"""Extract an audio segment from a video.

Use cases:
  - AI wants to feed a short audio chunk to Whisper for re-transcription
  - AI wants to view a spectrogram (use --spectrogram)
  - Quick listening test for human reviewer
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def audio_clip(
    src: Path,
    *,
    start: float,
    end: float,
    out: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    fmt: str | None = None,
) -> None:
    """Pull [start, end] of src as audio. Format inferred from out.suffix unless `fmt` set."""
    dur = end - start
    if dur <= 0:
        raise ValueError(f"end must be > start (got {start} {end})")
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or out.suffix.lstrip(".").lower() or "wav"

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
    ]
    if fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "128k"]
    elif fmt == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    elif fmt == "ogg":
        cmd += ["-c:a", "libvorbis"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True)


def spectrogram(
    src: Path,
    *,
    start: float,
    end: float,
    out: Path,
    width: int = 1024,
    height: int = 512,
) -> None:
    """Render a PNG spectrogram of the audio segment (for VLM to look at)."""
    dur = end - start
    if dur <= 0:
        raise ValueError(f"end must be > start")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
        "-lavfi", f"showspectrumpic=s={width}x{height}:legend=1",
        "-frames:v", "1",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def cli(src: Path, *, start: float, end: float, out: Path,
        sample_rate: int, mono: bool, with_spectrogram: bool) -> None:
    audio_clip(src, start=start, end=end, out=out, sample_rate=sample_rate, mono=mono)
    print(f"wrote {out}", file=sys.stderr)
    if with_spectrogram:
        spec_out = out.with_suffix(".spectrogram.png")
        spectrogram(src, start=start, end=end, out=spec_out)
        print(f"wrote {spec_out}", file=sys.stderr)
