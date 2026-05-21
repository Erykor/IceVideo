"""IceVideo CLI — a toolbox, not a pipeline.

Each subcommand is an independent tool. The AI calling these tools is the editor;
the tools just produce data or execute precise operations. No `auto` / `run-all`.

Commands by role:

  SENSORS    probe, extract, transcribe, frames, signals, clip-score, boundaries,
             telemetry, faces, diarize, music, audio-clip
  ANALYZERS  timeline, describe
  EFFECTORS  cut, concat, subtitle, music-mix
  ADVISORS   select, lint, critique, render (= cut+concat batch)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from icevideo import (
    audio_clip, boundaries, clip_score, concat, critique, cut, describe, diarize,
    extract, faces, frames, lint as lint_mod, music, music_mix, probe, render,
    select, signals, subtitle as subtitle_mod, telemetry, timeline, transcribe,
)
from icevideo.config import Paths, load_config


PER_VIDEO_DATA_STEPS = {
    "extract": extract.run,
    "transcribe": transcribe.run,
    "signals": signals.run,
    "clip-score": clip_score.run,
    "boundaries": boundaries.run,
    "telemetry": telemetry.run,
    "faces": faces.run,
    "diarize": diarize.run,
    "music": music.run,
}
ADVISORS = {
    "select": select.run,
    "lint": lint_mod.run,
    "critique": critique.run,
    "render": render.run,
}


def _build_paths(args, cfg: dict) -> Paths:
    paths = Paths.from_config(cfg, base=Path(args.cwd or "."))
    if getattr(args, "input_dir", None):
        dirs = args.input_dir if isinstance(args.input_dir, list) else [args.input_dir]
        resolved = [Path(d).expanduser().resolve() for d in dirs]
        paths.input_dir = resolved[0]
        paths.input_dirs = resolved
    if getattr(args, "work_dir", None):
        paths.work_dir = Path(args.work_dir).expanduser().resolve()
    if getattr(args, "output_dir", None):
        paths.output_dir = Path(args.output_dir).expanduser().resolve()
    if getattr(args, "input_glob", None):
        paths.input_glob = args.input_glob
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _add_project_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="Path to a config.toml override.")
    p.add_argument("--cwd", help="Working directory (default: current).")
    p.add_argument("--input-dir", action="append",
                   help="Input video directory (repeatable to merge multiple).")
    p.add_argument("--work-dir", help="Override [paths.work_dir].")
    p.add_argument("--output-dir", help="Override [paths.output_dir].")
    p.add_argument("--input-glob", help="Override [paths.input_glob], e.g. 'GX*.MP4'.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="icevideo",
        description="A toolbox for AI-driven video editing. Each tool is independent.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # data-producing steps that need project paths
    for name in [*PER_VIDEO_DATA_STEPS, *ADVISORS]:
        sp = sub.add_parser(name, help=f"({'data' if name in PER_VIDEO_DATA_STEPS else 'advisory'}) {name}")
        _add_project_flags(sp)

    # probe
    sp = sub.add_parser("probe", help="(sensor) Print video metadata (duration/codec/fps).")
    sp.add_argument("videos", nargs="+", type=Path)
    sp.add_argument("--json", action="store_true")

    # frames
    sp = sub.add_parser("frames", help="(sensor) Extract specific frames at specific timestamps.")
    sp.add_argument("video", type=Path)
    sp.add_argument("--at", type=float, nargs="+", help="Specific timestamps in seconds.")
    sp.add_argument("--every", help="Uniform sampling, e.g. '5s'.")
    sp.add_argument("--around", type=float, help="Center time for dense sampling.")
    sp.add_argument("--window", type=float, default=5.0, help="±seconds around --around.")
    sp.add_argument("--step", type=float, default=0.5, help="Sample step for --around.")
    sp.add_argument("--around-peaks", type=int, help="N frames around top peaks.")
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--scale", default="640:-1")
    sp.add_argument("--work-dir", type=Path,
                    help="For --around-peaks; defaults to ./work.")

    # timeline
    sp = sub.add_parser("timeline", help="(analyzer) Per-second readable timeline.")
    sp.add_argument("video", type=Path)
    sp.add_argument("--work-dir", type=Path)
    sp.add_argument("--csv", action="store_true")
    sp.add_argument("--only-active", action="store_true")
    sp.add_argument("--out", type=Path)

    # describe
    sp = sub.add_parser("describe", help="(analyzer) VLM caption for one or more frames.")
    sp.add_argument("images", nargs="+", type=Path)
    sp.add_argument("--prompt")
    sp.add_argument("--model")

    # cut
    sp = sub.add_parser("cut", help="(effector) Cut a single segment, or batch via --from-csv.")
    sp.add_argument("src", type=Path, nargs="?")
    sp.add_argument("--start", type=float)
    sp.add_argument("--end", type=float)
    sp.add_argument("--out", type=Path)
    sp.add_argument("--slow", type=float, help="Time-scale factor (e.g. 0.7 = 70% speed).")
    sp.add_argument("--interp", choices=["linear", "mci", "blend"], default="linear",
                    help="Slow-motion interpolation method.")
    sp.add_argument("--width", type=int, default=1920)
    sp.add_argument("--height", type=int, default=1080)
    sp.add_argument("--fps", type=int, default=30)
    sp.add_argument("--brightness", type=float, default=0.0)
    sp.add_argument("--saturation", type=float, default=1.0)
    sp.add_argument("--no-loudnorm", action="store_true")
    sp.add_argument("--codec")
    sp.add_argument("--audio-offset", type=float, default=0.0,
                    help="J cut: negative pulls audio earlier than video.")
    sp.add_argument("--snap-start", action="store_true",
                    help="Snap --start to nearest silence/scene from work/boundaries.")
    sp.add_argument("--snap-end", action="store_true")
    sp.add_argument("--work-dir", type=Path, help="For --snap-start/--snap-end (default ./work).")
    # batch mode
    sp.add_argument("--from-csv", help="CSV path or '-' for stdin (cols: src,start,end[,out,slow,interp,brightness,saturation,audio_offset,snap_start,snap_end,name]).")
    sp.add_argument("--out-dir", type=Path, help="Batch output directory (default ./output/clips).")
    sp.add_argument("--input-dir", type=Path, help="For batch mode: prefix to relative src paths.")

    # concat
    sp = sub.add_parser("concat", help="(effector) Chain pre-cut clips with xfade.")
    sp.add_argument("clips", nargs="+", type=Path)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--xfade", type=float, default=0.3)
    sp.add_argument("--audio-xfades", help="Per-seam audio crossfade durations, "
                    "e.g. '0.5,0.3,0.8' or sparse '1:0.5,3:0.8' for J/L cuts.")
    sp.add_argument("--codec")
    sp.add_argument("--cq", type=int, default=21)

    # subtitle
    sp = sub.add_parser("subtitle", help="(effector) Burn SRT into video (or soft-sub).")
    sp.add_argument("video", type=Path)
    sp.add_argument("srt", type=Path)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--softsub", action="store_true", help="Mux as soft subtitle track instead of burning.")
    sp.add_argument("--font-size", type=int, default=28)
    sp.add_argument("--alignment", type=int, default=2, help="ASS numpad alignment, 2 = bottom-center.")
    sp.add_argument("--margin-v", type=int, default=30)
    sp.add_argument("--codec")
    sp.add_argument("--cq", type=int, default=23)

    # music-mix
    sp = sub.add_parser("music-mix", help="(effector) Add music bed with ducking.")
    sp.add_argument("video", type=Path)
    sp.add_argument("music", type=Path)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--music-volume-db", type=float, default=-8.0)
    sp.add_argument("--duck-db", type=float, default=-10.0)
    sp.add_argument("--no-loop", action="store_true", help="Don't loop the music if shorter than video.")
    sp.add_argument("--codec")
    sp.add_argument("--cq", type=int, default=21)

    # audio-clip
    sp = sub.add_parser("audio-clip", help="(sensor) Extract audio segment (with optional spectrogram).")
    sp.add_argument("src", type=Path)
    sp.add_argument("--start", type=float, required=True)
    sp.add_argument("--end", type=float, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--sample-rate", type=int, default=16000)
    sp.add_argument("--stereo", action="store_true")
    sp.add_argument("--spectrogram", action="store_true",
                    help="Also write <out>.spectrogram.png for VLM inspection.")

    # info
    sp = sub.add_parser("info", help="Print resolved paths/config.")
    _add_project_flags(sp)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "probe":
        probe.cli(args.videos, as_json=args.json); return 0

    if args.cmd == "describe":
        describe.cli(args.images, prompt=getattr(args, "prompt", None),
                     model=getattr(args, "model", None))
        return 0

    if args.cmd == "frames":
        work_dir = args.work_dir or Path("./work")
        frames.cli(args.video, at=args.at, every=args.every, around=args.around,
                   window=args.window, step=args.step, around_peaks=args.around_peaks,
                   out_dir=args.out, scale=args.scale, work_dir=work_dir)
        return 0

    if args.cmd == "timeline":
        work_dir = args.work_dir or Path("./work")
        timeline.cli(args.video, work_dir, fmt="csv" if args.csv else "md",
                     only_active=args.only_active, out=args.out)
        return 0

    if args.cmd == "cut":
        if args.from_csv:
            args.work_dir = args.work_dir or Path("./work")
            cut.cli(args); return 0
        # single mode
        for need in ("src", "start", "end", "out"):
            if getattr(args, need) is None:
                raise SystemExit(f"cut: --{need.replace('_','-')} is required in single mode")
        args.work_dir = args.work_dir or Path("./work")
        cut.cli(args)
        return 0

    if args.cmd == "concat":
        concat.cli(args); return 0

    if args.cmd == "subtitle":
        subtitle_mod.cli(args); return 0

    if args.cmd == "music-mix":
        music_mix.cli(args); return 0

    if args.cmd == "audio-clip":
        audio_clip.cli(args.src, start=args.start, end=args.end, out=args.out,
                       sample_rate=args.sample_rate, mono=not args.stereo,
                       with_spectrogram=args.spectrogram)
        return 0

    # Project-path commands.
    for attr in ("config", "cwd", "input_dir", "work_dir", "output_dir", "input_glob"):
        if not hasattr(args, attr):
            setattr(args, attr, None)
    cfg = load_config(args.config, cwd=args.cwd)

    if args.cmd == "info":
        paths = _build_paths(args, cfg)
        print(f"input_dirs  = {paths.input_dirs}")
        print(f"work_dir    = {paths.work_dir}")
        print(f"output_dir  = {paths.output_dir}")
        print(f"input_glob  = {paths.input_glob}")
        return 0

    paths = _build_paths(args, cfg)
    fn = PER_VIDEO_DATA_STEPS.get(args.cmd) or ADVISORS.get(args.cmd)
    if fn is None:
        raise SystemExit(f"unknown command: {args.cmd}")
    fn(paths, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
