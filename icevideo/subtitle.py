"""Burn subtitles into a video. Critical for muted-by-default platforms (TikTok/Reels).

Two modes:
  burned (default): re-encode video with subtitles "painted" into pixels.
  softsub: add subtitle as a selectable track (smaller file, but player must support).
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


def burn(
    video: Path,
    srt: Path,
    out: Path,
    *,
    font_size: int = 28,
    font_color: str = "&HFFFFFF&",         # white (ASS BGR hex)
    outline_color: str = "&H000000&",      # black outline
    outline_thickness: int = 2,
    alignment: int = 2,                     # 2 = bottom-center (ASS numpad)
    margin_v: int = 30,
    codec: str | None = None,
    cq: int = 23,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    codec = codec or ("h264_nvenc" if _has_nvenc() else "libx264")
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"
    preset = "p4" if codec.endswith("_nvenc") else "veryfast"

    # The ffmpeg subtitles filter wants forward slashes even on Windows; for paths with
    # spaces we wrap them with quotes and escape colons (`subtitles=…` parser is fussy).
    srt_arg = str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
    style = (
        f"FontSize={font_size},PrimaryColour={font_color},"
        f"OutlineColour={outline_color},Outline={outline_thickness},"
        f"BorderStyle=1,Alignment={alignment},MarginV={margin_v}"
    )
    vf = f"subtitles='{srt_arg}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf", vf,
        "-c:v", codec, "-preset", preset, qarg, str(cq),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def softsub(video: Path, srt: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-i", str(srt),
        "-c", "copy", "-c:s", "mov_text",
        "-map", "0:v", "-map", "0:a", "-map", "1:s",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def cli(args) -> None:
    if args.softsub:
        softsub(args.video, args.srt, args.out)
    else:
        burn(args.video, args.srt, args.out,
             font_size=args.font_size, alignment=args.alignment,
             margin_v=args.margin_v, codec=args.codec, cq=args.cq)
    print(f"wrote {args.out}", file=sys.stderr)
