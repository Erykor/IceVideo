"""Extract specific frames from a video — for the AI to view via Read.

Three modes:
  --at 12.5 27.0 ...    # extract at exact timestamps
  --every 5s            # uniformly every N seconds
  --around-peaks N      # N frames around each peak in work/signals (top-K by combined score)

Output names encode the timestamp so the AI can correlate frames with the timeline:
  <out_dir>/<base>_t012.50.jpg
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from icevideo.utils import log, probe_duration, video_basename


def extract_at(video: Path, times: list[float], out_dir: Path,
               *, scale: str = "640:-1", quality: int = 4) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = video_basename(video)
    paths: list[Path] = []
    for t in times:
        if t < 0:
            continue
        out = out_dir / f"{base}_t{t:07.2f}.jpg"
        if out.exists():
            paths.append(out); continue
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={scale}", "-q:v", str(quality),
            str(out),
        ], check=True)
        paths.append(out)
    return paths


def _every(dur: float, step: float) -> list[float]:
    times: list[float] = []
    t = step / 2
    while t < dur:
        times.append(round(t, 2))
        t += step
    return times


def _parse_interval(spec: str) -> float:
    """'5s' -> 5.0; '0.5s' -> 0.5; '5' -> 5.0"""
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*s?$", spec.strip())
    if not m:
        raise ValueError(f"bad interval: {spec}")
    return float(m.group(1))


def _peak_times(signals_json: Path, n: int) -> list[float]:
    with open(signals_json) as f:
        d = json.load(f)
    motion = d.get("scene", [])
    flow = d.get("flow", [])
    n_sec = d.get("seconds", 0)
    if not motion or not flow:
        return []
    scores = [motion[i] + 0.1 * flow[i] for i in range(min(len(motion), len(flow), n_sec))]
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [float(i) for i in sorted(ranked[:n])]


def _around(center: float, window: float, step: float, dur: float) -> list[float]:
    """± window seconds around `center`, every `step` seconds, clipped to [0, dur]."""
    t = max(0.0, center - window)
    end = min(dur, center + window)
    out: list[float] = []
    while t <= end + 1e-6:
        out.append(round(t, 2))
        t += step
    return out


def cli(video: Path, *, at: list[float] | None, every: str | None, around_peaks: int | None,
        around: float | None, window: float, step: float,
        out_dir: Path, scale: str = "640:-1", work_dir: Path | None = None) -> None:
    dur = probe_duration(video)
    times: list[float] = []
    if at:
        times = list(at)
    elif every:
        times = _every(dur, _parse_interval(every))
    elif around is not None:
        times = _around(around, window, step, dur)
    elif around_peaks:
        if work_dir is None:
            raise SystemExit("--around-peaks needs --work-dir to read signals")
        sigp = work_dir / "signals" / f"{video.stem}.json"
        if not sigp.exists():
            raise SystemExit(f"no signals for {video.stem}; run `icevideo signals` first")
        times = _peak_times(sigp, around_peaks)
    else:
        raise SystemExit("must supply one of --at / --every / --around / --around-peaks")

    paths = extract_at(video, times, out_dir, scale=scale)
    for p in paths:
        print(p)
    log(f"{video.stem}: {len(paths)} frames in {out_dir}", "frames")
