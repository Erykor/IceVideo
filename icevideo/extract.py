"""Step 1: extract audio + CLIP frames + sample frames per source video."""
from __future__ import annotations

from pathlib import Path

from icevideo.config import Paths, discover_videos
from icevideo.utils import log, probe_duration, run_ffmpeg, video_basename


def extract_audio(video: Path, dst: Path) -> None:
    """16 kHz mono PCM WAV for Whisper + audio analysis."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dst),
    ])


def extract_clip_frames(video: Path, dst_dir: Path, fps: float) -> int:
    """0.5 fps 224x224 frames used by the CLIP scoring step."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.glob("*.jpg")):
        return len(list(dst_dir.glob("*.jpg")))
    run_ffmpeg([
        "-y", "-i", str(video),
        "-vf", f"fps={fps},scale=224:224:flags=lanczos",
        str(dst_dir / "%04d.jpg"),
    ])
    return len(list(dst_dir.glob("*.jpg")))


def extract_sample_frames(video: Path, dst_dir: Path, count: int) -> None:
    """A few evenly-spaced JPEGs per video so humans can spot-check coverage."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = video_basename(video)
    if any(dst_dir.glob(f"{base}_*.jpg")):
        return
    dur = probe_duration(video)
    for i in range(1, count + 1):
        t = dur * i / (count + 1)
        run_ffmpeg([
            "-y", "-ss", f"{t:.2f}", "-i", str(video),
            "-frames:v", "1", "-vf", "scale=480:-1",
            str(dst_dir / f"{base}_{i}.jpg"),
        ])


def run(paths: Paths, cfg: dict) -> None:
    sig = cfg["signals"]
    audio_dir = paths.subdir("audio")
    clip_dir = paths.subdir("clip_frames")
    sample_dir = paths.subdir("frames")
    videos = discover_videos(paths)
    if not videos:
        log("no videos matched input_glob", "extract")
        return
    for v in videos:
        base = video_basename(v)
        extract_audio(v, audio_dir / f"{base}.wav")
        n = extract_clip_frames(v, clip_dir / base, sig["clip_fps"])
        extract_sample_frames(v, sample_dir, sig["sample_frames"])
        log(f"{base}: WAV + {n} CLIP frames", "extract")
