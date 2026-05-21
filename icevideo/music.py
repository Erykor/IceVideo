"""Step 5.7: optional music bed + onset-snapped cuts.

If `[render.music_path]` (or env var ICEVIDEO_MUSIC) points to an audio file, we:
  - run librosa onset detection on it
  - write work/music/onsets.json with beat times
  - render.py will then mix the music as a background bed
  - select.py (if music is enabled) snaps segment durations to be multiples of beat-intervals

If no music is provided, this step is a no-op.
"""
from __future__ import annotations

import os
from pathlib import Path

from icevideo.config import Paths
from icevideo.utils import log, save_json


def detect_onsets(audio_path: Path) -> dict:
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    y, sr = librosa.load(str(audio_path), mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time").tolist()
    duration = float(len(y) / sr)
    # `tempo` may be a 1-element ndarray on newer librosa; coerce to float
    tempo_val = float(tempo) if np.ndim(tempo) == 0 else float(np.asarray(tempo).flat[0])
    return {
        "audio": str(audio_path),
        "duration": duration,
        "tempo_bpm": tempo_val,
        "beat_times": beat_times,
        "onsets": onsets,
    }


def run(paths: Paths, cfg: dict) -> None:
    music_path = cfg.get("render", {}).get("music_path") or os.environ.get("ICEVIDEO_MUSIC")
    if not music_path:
        log("no music_path configured; skipping", "music")
        save_json(paths.subdir("music") / "onsets.json", {"skipped": True})
        return
    music_path = Path(music_path).expanduser().resolve()
    if not music_path.exists():
        log(f"music file not found: {music_path}", "music")
        save_json(paths.subdir("music") / "onsets.json", {"skipped": True, "error": "not_found"})
        return
    try:
        data = detect_onsets(music_path)
        save_json(paths.subdir("music") / "onsets.json", data)
        log(f"tempo={data['tempo_bpm']:.1f} BPM  beats={len(data['beat_times'])}  duration={data['duration']:.1f}s",
            "music")
    except ImportError:
        log("librosa not installed; install with `uv add librosa`; skipping", "music")
        save_json(paths.subdir("music") / "onsets.json", {"skipped": True, "error": "no_librosa"})
    except Exception as e:
        log(f"onset detection failed — {e}", "music")
        save_json(paths.subdir("music") / "onsets.json", {"skipped": True, "error": str(e)})
