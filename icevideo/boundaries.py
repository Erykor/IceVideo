"""Step 5: extract silence intervals and scene cut timestamps for boundary snapping."""
from __future__ import annotations

import re
from pathlib import Path

from icevideo.config import Paths, discover_videos
from icevideo.utils import log, run_ffmpeg_with_stderr, save_json, video_basename


def detect_silence(video: Path, noise_db: float, min_dur: float) -> list[tuple[float, float]]:
    stderr = run_ffmpeg_with_stderr([
        "-i", str(video),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ])
    intervals: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in stderr.splitlines():
        m = re.search(r"silence_start: ([0-9.]+)", line)
        if m:
            cur_start = float(m.group(1)); continue
        m = re.search(r"silence_end: ([0-9.]+).*silence_duration: ([0-9.]+)", line)
        if m and cur_start is not None:
            intervals.append((cur_start, float(m.group(1))))
            cur_start = None
    return intervals


def detect_scenes(video: Path, thr: float) -> list[float]:
    stderr = run_ffmpeg_with_stderr([
        "-i", str(video),
        "-vf", f"select='gt(scene,{thr})',metadata=print",
        "-an", "-f", "null", "-",
    ])
    return [float(m.group(1)) for m in re.finditer(r"pts_time:([0-9.]+)", stderr)]


def run(paths: Paths, cfg: dict) -> None:
    bcfg = cfg["boundaries"]
    out_dir = paths.subdir("boundaries")

    for video in discover_videos(paths):
        base = video_basename(video)
        out_json = out_dir / f"{base}.json"
        if out_json.exists():
            log(f"{base}: skip", "boundaries")
            continue
        silence = detect_silence(video, bcfg["silence_noise_db"], bcfg["silence_min_dur"])
        scenes = detect_scenes(video, bcfg["scene_threshold"])
        save_json(out_json, {"video": base, "silence": silence, "scenes": scenes})
        log(f"{base}: silence={len(silence)} scenes={len(scenes)}", "boundaries")
