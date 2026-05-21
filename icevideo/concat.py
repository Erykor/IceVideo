"""Chain pre-cut clips into a single output.

Standard xfade chain plus optional cross-segment J/L cuts: each seam can have its own
audio crossfade duration, letting next-clip audio bleed into previous-clip video (J)
or previous-clip audio bleed into next-clip video (L).

The simplest, robust implementation: each seam has an `audio_xfade` (how long the
audio crossfade lasts) that can be > or < the video `xfade` duration. When audio
crossfade is *longer* than video xfade you get a J cut (audio of next clip starts
before its video). The crossfade still ends together with the video transition.
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


def _probe_dur(p: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(p),
    ]).strip())


def concat(
    clips: list[Path],
    out: Path,
    *,
    xfade: float = 0.3,
    audio_xfades: list[float] | None = None,
    codec: str | None = None,
    cq: int = 21,
    audio_bitrate: str = "192k",
) -> None:
    """Chain `clips` into `out` using xfade.

    `audio_xfades[i]` is the audio crossfade duration at the seam between clip i and
    clip i+1. If None, defaults to `xfade` everywhere. Setting it longer than `xfade`
    yields J-cut effect (next clip's audio bleeds earlier).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if not clips:
        raise ValueError("need at least 1 clip to concat")
    if len(clips) == 1:
        shutil.copy(clips[0], out); return

    codec = codec or ("h264_nvenc" if _has_nvenc() else "libx264")
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"
    preset = "p4" if codec.endswith("_nvenc") else "veryfast"

    if xfade <= 0:
        listfile = out.parent / f".{out.stem}_concat.txt"
        listfile.write_text("\n".join(f"file '{p}'" for p in clips))
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-c", "copy", "-movflags", "+faststart", str(out),
        ], check=True)
        listfile.unlink()
        return

    durs = [_probe_dur(p) for p in clips]
    n_seams = len(clips) - 1
    if audio_xfades is None:
        audio_xfades = [xfade] * n_seams
    if len(audio_xfades) != n_seams:
        raise ValueError(f"audio_xfades must have {n_seams} entries (one per seam), got {len(audio_xfades)}")

    inputs: list[str] = []
    for p in clips:
        inputs += ["-i", str(p)]

    fc_lines: list[str] = []
    prev_v = "[0:v]"
    cum = 0.0
    for i in range(1, len(clips)):
        cum += durs[i - 1] - xfade
        out_v = f"[v{i}]"
        fc_lines.append(f"{prev_v}[{i}:v]xfade=transition=fade:duration={xfade}:offset={cum:.3f}{out_v}")
        prev_v = out_v

    prev_a = "[0:a]"
    for i in range(1, len(clips)):
        out_a = f"[a{i}]"
        afade = audio_xfades[i - 1]
        fc_lines.append(f"{prev_a}[{i}:a]acrossfade=d={afade}{out_a}")
        prev_a = out_a

    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", ";".join(fc_lines),
        "-map", prev_v, "-map", prev_a,
        "-c:v", codec, "-preset", preset, qarg, str(cq),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(out),
    ], check=True)


def _parse_audio_xfades(spec: str | None, n_seams: int, default: float) -> list[float] | None:
    """Parse e.g. "0.5,0.3,0.8" or "1:0.5,3:0.8" -> per-seam list of length n_seams.
    The latter form is sparse: only the listed seam indices get an override.
    """
    if spec is None or not spec.strip():
        return None
    out = [default] * n_seams
    items = spec.split(",")
    if ":" in items[0]:
        for it in items:
            k, v = it.split(":")
            i = int(k)
            if not 0 <= i < n_seams:
                raise ValueError(f"audio-xfade seam index {i} out of range [0, {n_seams})")
            out[i] = float(v)
    else:
        for i, it in enumerate(items[:n_seams]):
            out[i] = float(it)
    return out


def cli(args) -> None:
    n_seams = max(0, len(args.clips) - 1)
    afx = _parse_audio_xfades(args.audio_xfades, n_seams, args.xfade)
    concat(args.clips, out=args.out, xfade=args.xfade,
           audio_xfades=afx, codec=args.codec, cq=args.cq)
    dur = _probe_dur(args.out)
    print(f"wrote {args.out}  ({dur:.1f}s)", file=sys.stderr)
