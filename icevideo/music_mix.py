"""Apply a music bed under an existing video, ducking the original audio.

Standalone — you can take any rendered video and add music later without re-cutting.

Implementation uses sidechain compression so the music dips when the original audio
is loud (typical for dialogue-heavy clips), then amix to combine.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _has_nvenc() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
        return "h264_nvenc" in out
    except Exception:
        return False


def music_mix(
    video_in: Path,
    music: Path,
    video_out: Path,
    *,
    music_volume_db: float = -8.0,
    duck_db: float = -10.0,       # how much the music drops when speech is detected
    duck_attack_ms: int = 10,
    duck_release_ms: int = 300,
    duck_threshold: float = 0.05,
    music_loop: bool = True,
    codec: str | None = None,
    cq: int = 21,
    audio_bitrate: str = "192k",
) -> None:
    """Layer music_in under video_in's audio with sidechain ducking."""
    video_out.parent.mkdir(parents=True, exist_ok=True)
    loop_flag = "aloop=loop=-1:size=2e9," if music_loop else ""
    codec = codec or ("h264_nvenc" if _has_nvenc() else "libx264")
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"
    preset = "p4" if codec.endswith("_nvenc") else "veryfast"

    # Note: sidechaincompress reduces music when [side] is loud. `duck_db` is a target
    # gain reduction; convert to ratio for the compressor (loose mapping).
    duck_ratio = max(2, min(20, int(abs(duck_db) / 2)))

    filtergraph = (
        f"[1:a]{loop_flag}volume={music_volume_db}dB,asetpts=N/SR/TB[mus];"
        "[0:a]asplit=2[orig1][orig_for_side];"
        "[orig_for_side]aformat=channel_layouts=stereo,asetpts=N/SR/TB[side];"
        f"[mus][side]sidechaincompress=threshold={duck_threshold}:ratio={duck_ratio}:"
        f"attack={duck_attack_ms}:release={duck_release_ms}[ducked];"
        "[orig1][ducked]amix=inputs=2:duration=first:weights=1 1[mix]"
    )

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_in), "-i", str(music),
        "-filter_complex", filtergraph,
        "-map", "0:v", "-map", "[mix]",
        "-c:v", "copy",                                # keep video stream as-is
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-shortest",
        "-movflags", "+faststart",
        str(video_out),
    ]
    subprocess.run(cmd, check=True)


def cli(args) -> None:
    music_mix(
        args.video, args.music, args.out,
        music_volume_db=args.music_volume_db, duck_db=args.duck_db,
        music_loop=not args.no_loop,
        codec=args.codec, cq=args.cq,
    )
    print(f"wrote {args.out}", file=sys.stderr)
