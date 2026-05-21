"""Cut a single segment out of one source video. The AI calls this once per segment.

Supports:
  * normal cut: --start --end --out
  * slow motion: --slow 0.7   (with optional --interp mci|blend|linear)
  * loudness normalization (default on)
  * brightness + saturation tweaks
  * J cut within the clip: --audio-offset -0.5
  * snap start/end to boundary timestamps:  --snap-start --snap-end --work-dir
  * batch mode: --from-csv FILE.csv --out-dir DIR/
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def _has_nvenc() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
        return "h264_nvenc" in out
    except Exception:
        return False


def _loudnorm_pass1(src: Path, start: float, dur: float,
                    target_i: float, target_tp: float, target_lra: float) -> dict | None:
    proc = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
        "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
        "-f", "null", "-",
    ], capture_output=True, text=True)
    m = re.search(r"(\{[^{]*\"input_i\".*?\})", proc.stderr, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _snap_to_boundary(t: float, boundaries: dict, *, kind: str, window: float = 0.7) -> float:
    """Snap t to the nearest silence midpoint or scene-cut within `window` seconds.
    Scene cuts within 0.4s win; otherwise pick the closest of any kind.
    """
    silence = boundaries.get("silence", []) or []
    scenes = boundaries.get("scenes", []) or []
    cands: list[tuple[str, float, float]] = []
    for ss, se in silence:
        mid = (ss + se) / 2
        if abs(mid - t) <= window:
            cands.append(("silence", mid, abs(mid - t)))
    for sc in scenes:
        if abs(sc - t) <= window:
            cands.append(("scene", sc, abs(sc - t)))
    if not cands:
        return t
    near_scenes = [c for c in cands if c[0] == "scene" and c[2] <= 0.4]
    if near_scenes:
        return min(near_scenes, key=lambda c: c[2])[1]
    return min(cands, key=lambda c: c[2])[1]


def _load_boundaries(src: Path, work_dir: Path) -> dict | None:
    p = work_dir / "boundaries" / f"{src.stem}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def cut(
    src: Path,
    *,
    start: float,
    end: float,
    out: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    slow: float | None = None,
    interp: str = "linear",            # "linear" (setpts only) | "mci" (motion-compensated) | "blend"
    interp_fps: int = 60,
    loudnorm: bool = True,
    loudnorm_i: float = -16.0,
    loudnorm_tp: float = -1.5,
    loudnorm_lra: float = 11.0,
    audio_fade: float = 0.12,
    brightness: float = 0.0,
    saturation: float = 1.0,
    codec: str | None = None,
    preset: str | None = None,
    cq: int = 23,
    audio_bitrate: str = "192k",
    audio_offset: float = 0.0,
    snap_start: bool = False,
    snap_end: bool = False,
    work_dir: Path | None = None,
    snap_window: float = 0.7,
) -> tuple[float, float]:
    """Cut [start, end] from src to out. Returns the (final_start, final_end) after snapping.
    `snap_start` / `snap_end` need a `work_dir` with `boundaries/<stem>.json` produced by
    `icevideo boundaries` — silently fall back to the requested values if absent.
    """
    if snap_start or snap_end:
        bnds = _load_boundaries(src, work_dir or Path("./work"))
        if bnds:
            if snap_start:
                start = _snap_to_boundary(start, bnds, kind="start", window=snap_window)
            if snap_end:
                end = _snap_to_boundary(end, bnds, kind="end", window=snap_window)

    dur = end - start
    if dur <= 0:
        raise ValueError(f"end must be > start (got {start} {end})")
    out.parent.mkdir(parents=True, exist_ok=True)

    fade = min(audio_fade, dur / 4)
    codec = codec or ("h264_nvenc" if _has_nvenc() else "libx264")
    preset = preset or ("p4" if codec.endswith("_nvenc") else "veryfast")
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"

    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
    ]
    if slow:
        if interp == "mci":
            # motion-compensated interpolation, then time-stretch
            vf_parts.append(f"minterpolate=fps={interp_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
            vf_parts.append(f"setpts=PTS/{slow}")
        elif interp == "blend":
            vf_parts.append(f"minterpolate=fps={interp_fps}:mi_mode=blend")
            vf_parts.append(f"setpts=PTS/{slow}")
        else:  # linear / no interpolation
            vf_parts.append(f"setpts=PTS/{slow}")
    vf_parts.append(f"fps={fps}")
    if abs(brightness) > 0.005 or abs(saturation - 1.0) > 0.005:
        b = max(-0.15, min(0.15, brightness))
        vf_parts.append(f"eq=brightness={b:.3f}:saturation={saturation:.2f}")
    vf = ",".join(vf_parts)

    out_dur = dur / slow if slow else dur
    af_parts: list[str] = []
    if loudnorm:
        meas = _loudnorm_pass1(src, start, dur, loudnorm_i, loudnorm_tp, loudnorm_lra)
        if meas:
            af_parts.append(
                f"loudnorm=I={loudnorm_i}:TP={loudnorm_tp}:LRA={loudnorm_lra}:"
                f"measured_I={meas['input_i']}:measured_LRA={meas['input_lra']}:"
                f"measured_TP={meas['input_tp']}:measured_thresh={meas['input_thresh']}:"
                f"offset={meas['target_offset']}:linear=true:print_format=summary"
            )
        else:
            af_parts.append(f"loudnorm=I={loudnorm_i}:TP={loudnorm_tp}:LRA={loudnorm_lra}")
    af_parts.append(f"afade=t=in:st=0:d={fade}")
    if slow:
        af_parts.append(f"atempo={slow}")
    af_parts.append(f"afade=t=out:st={out_dur - fade:.3f}:d={fade}")

    audio_ss = start
    audio_dur = dur
    if audio_offset:
        audio_ss = max(0.0, start + audio_offset)
        # Audio duration MUST match video duration; the offset shifts WHEN the
        # audio is sampled from, not how much. Otherwise the container takes the
        # longer of the two streams and you get a duration mismatch.
        audio_dur = dur

    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if audio_offset != 0.0:
        cmd += [
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
            "-ss", f"{audio_ss:.3f}", "-t", f"{audio_dur:.3f}", "-i", str(src),
            "-map", "0:v", "-map", "1:a",
        ]
    else:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
    cmd += [
        "-vf", vf, "-af", ",".join(af_parts),
        "-c:v", codec, "-preset", preset, qarg, str(cq),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return start, end


# ---------------------------------------------------------------------------
# Batch mode: read a CSV (or stdin) of cut specs, run them all.
# ---------------------------------------------------------------------------

REQUIRED_BATCH_COLS = {"src", "start", "end"}
OPTIONAL_BATCH_COLS = {
    "out", "slow", "interp", "brightness", "saturation", "audio_offset",
    "snap_start", "snap_end", "name",
}


def _coerce(v: str, kind: str):
    if v == "" or v is None:
        return None
    if kind == "float":
        return float(v)
    if kind == "bool":
        return v.lower() in ("1", "true", "yes", "y")
    return v


def cut_batch(
    rows: list[dict],
    *,
    out_dir: Path,
    input_dir: Path | None = None,
    work_dir: Path | None = None,
    default_width: int = 1920,
    default_height: int = 1080,
    default_fps: int = 30,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, r in enumerate(rows):
        miss = REQUIRED_BATCH_COLS - {k for k, v in r.items() if v not in (None, "")}
        if miss:
            raise SystemExit(f"row {i}: missing required columns {sorted(miss)}")
        src = Path(r["src"])
        if not src.is_absolute() and input_dir:
            src = input_dir / src
        start = float(r["start"]); end = float(r["end"])
        name = r.get("name") or f"{i:03d}_{src.stem}_{start:.2f}-{end:.2f}.mp4"
        if not name.endswith(".mp4"):
            name += ".mp4"
        out = Path(r["out"]) if r.get("out") else (out_dir / name)
        cut(
            src,
            start=start, end=end, out=out,
            width=default_width, height=default_height, fps=default_fps,
            slow=_coerce(r.get("slow", ""), "float"),
            interp=r.get("interp") or "linear",
            brightness=_coerce(r.get("brightness", ""), "float") or 0.0,
            saturation=_coerce(r.get("saturation", ""), "float") or 1.0,
            audio_offset=_coerce(r.get("audio_offset", ""), "float") or 0.0,
            snap_start=_coerce(r.get("snap_start", ""), "bool") or False,
            snap_end=_coerce(r.get("snap_end", ""), "bool") or False,
            work_dir=work_dir,
        )
        written.append(out)
    return written


def _read_csv(path: str) -> list[dict]:
    if path == "-":
        return list(csv.DictReader(sys.stdin))
    with open(path) as f:
        return list(csv.DictReader(f))


def cli(args) -> None:
    if args.from_csv:
        rows = _read_csv(args.from_csv)
        out_dir = args.out_dir or Path("./output/clips")
        paths = cut_batch(
            rows,
            out_dir=out_dir,
            input_dir=args.input_dir,
            work_dir=args.work_dir,
            default_width=args.width, default_height=args.height, default_fps=args.fps,
        )
        for p in paths:
            print(p)
        return

    # single mode
    s, e = cut(
        args.src,
        start=args.start, end=args.end, out=args.out,
        slow=args.slow, interp=args.interp,
        width=args.width, height=args.height, fps=args.fps,
        brightness=args.brightness, saturation=args.saturation,
        loudnorm=not args.no_loudnorm, codec=args.codec,
        audio_offset=args.audio_offset,
        snap_start=args.snap_start, snap_end=args.snap_end,
        work_dir=args.work_dir,
    )
    if args.snap_start or args.snap_end:
        print(f"final start/end: {s:.3f} {e:.3f}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
