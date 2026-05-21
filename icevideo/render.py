"""Step 7: NVENC/x264 render with loudnorm, brightness normalization, speed ramp and xfade."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from icevideo.config import Paths
from icevideo.utils import load_json, log


def _ffmpeg(args: list[str], *, capture: bool = False, check: bool = True):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *args]
    return subprocess.run(cmd, capture_output=capture, text=capture, check=check)


def _ffmpeg_info(args: list[str]) -> str:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", *args]
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def _measure_brightness(src: Path, start: float, dur: float) -> float:
    stderr = _ffmpeg_info([
        "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{min(dur, 2):.2f}",
        "-vf", "scale=320:180,signalstats,metadata=mode=print",
        "-an", "-f", "null", "-",
    ])
    vals = [float(m.group(1)) for m in re.finditer(r"lavfi\.signalstats\.YAVG=([0-9.]+)", stderr)]
    return float(sum(vals) / len(vals)) if vals else 128.0


def _loudnorm_pass1(src: Path, start: float, dur: float, target: dict) -> dict | None:
    stderr = _ffmpeg_info([
        "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{dur:.2f}",
        "-af", f"loudnorm=I={target['I']}:TP={target['TP']}:LRA={target['LRA']}:print_format=json",
        "-f", "null", "-",
    ])
    m = re.search(r"(\{[^{]*\"input_i\".*?\})", stderr, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _cut(src: Path, start: float, end: float, dst: Path, *,
         slow: bool, target_bright: float, seg_bright: float, cfg: dict) -> None:
    r = cfg["render"]
    dur = end - start
    fade = min(r["audio_fade"], dur / 4)

    br_delta = (target_bright - seg_bright) / 255.0
    cap = r["brightness_clamp"]
    br_delta = max(-cap, min(cap, br_delta))

    vf_parts = [
        f"scale={r['width']}:{r['height']}:force_original_aspect_ratio=decrease",
        f"pad={r['width']}:{r['height']}:(ow-iw)/2:(oh-ih)/2",
    ]
    if slow:
        vf_parts.append(f"setpts=PTS/{r['slow_factor']}")
    vf_parts.append(f"fps={r['fps']}")
    if abs(br_delta) > 0.005:
        vf_parts.append(f"eq=brightness={br_delta:.3f}:saturation=1.05")
    vf = ",".join(vf_parts)

    loud_target = {"I": r["loudnorm_i"], "TP": r["loudnorm_tp"], "LRA": r["loudnorm_lra"]}
    meas = _loudnorm_pass1(src, start, dur, loud_target)
    if meas:
        af_loud = (
            f"loudnorm=I={loud_target['I']}:TP={loud_target['TP']}:LRA={loud_target['LRA']}:"
            f"measured_I={meas['input_i']}:measured_LRA={meas['input_lra']}:"
            f"measured_TP={meas['input_tp']}:measured_thresh={meas['input_thresh']}:"
            f"offset={meas['target_offset']}:linear=true:print_format=summary"
        )
    else:
        af_loud = f"loudnorm=I={loud_target['I']}:TP={loud_target['TP']}:LRA={loud_target['LRA']}"

    out_dur = dur / r["slow_factor"] if slow else dur
    af_parts = [af_loud, f"afade=t=in:st=0:d={fade}"]
    if slow:
        af_parts.append(f"atempo={r['slow_factor']}")
    af_parts.append(f"afade=t=out:st={out_dur - fade:.3f}:d={fade}")
    af = ",".join(af_parts)

    codec = r["codec"]
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"
    _ffmpeg([
        "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-vf", vf, "-af", af,
        "-c:v", codec, "-preset", r["preset"], qarg, str(r["cq"]),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", r["audio_bitrate"], "-ac", "2", "-ar", "48000",
        "-movflags", "+faststart",
        str(dst),
    ])


def _xfade_chain(clip_paths: list[Path], dst: Path, cfg: dict) -> None:
    r = cfg["render"]
    if len(clip_paths) == 1:
        shutil.copy(clip_paths[0], dst); return

    durs = []
    for p in clip_paths:
        d = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(p),
        ]).strip())
        durs.append(d)

    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    n = len(clip_paths)
    xfade = r["xfade_duration"]
    fc_lines: list[str] = []
    prev_v = "[0:v]"
    cum = 0.0
    for i in range(1, n):
        cum += durs[i - 1] - xfade
        out_v = f"[v{i}]"
        fc_lines.append(f"{prev_v}[{i}:v]xfade=transition=fade:duration={xfade}:offset={cum:.3f}{out_v}")
        prev_v = out_v
    prev_a = "[0:a]"
    for i in range(1, n):
        out_a = f"[a{i}]"
        fc_lines.append(f"{prev_a}[{i}:a]acrossfade=d={xfade}{out_a}")
        prev_a = out_a
    fc = ";".join(fc_lines)

    codec = r["codec"]
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"
    _ffmpeg([
        "-y", *inputs,
        "-filter_complex", fc,
        "-map", prev_v, "-map", prev_a,
        "-c:v", codec, "-preset", r["preset"], qarg, str(max(r["cq"] - 2, 18)),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", r["audio_bitrate"],
        "-movflags", "+faststart",
        str(dst),
    ])


def _mix_music_bed(video_in: Path, music_path: Path, video_out: Path, cfg: dict) -> None:
    """Layer music under the existing audio, ducking original by `music_duck_db`."""
    r = cfg["render"]
    duck = r.get("music_duck_db", -10)
    music_vol_db = r.get("music_volume_db", -8)
    # use sidechaincompress for ducking: when speech is present, dip music
    # simpler approach: amix with weights + reduce music to -8dB and original kept
    codec = r["codec"]
    qarg = "-cq" if codec.endswith("_nvenc") else "-crf"
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_in), "-i", str(music_path),
        "-filter_complex", (
            f"[1:a]aloop=loop=-1:size=2e9,volume={music_vol_db}dB,"
            f"asetpts=N/SR/TB[mus];"
            "[0:a]asplit=2[orig1][orig_for_side];"
            "[orig_for_side]aformat=channel_layouts=stereo,asetpts=N/SR/TB[side];"
            "[mus][side]sidechaincompress=threshold=0.05:ratio=4:attack=10:release=300[ducked];"
            "[orig1][ducked]amix=inputs=2:duration=first:weights=1 1[mix]"
        ),
        "-map", "0:v", "-map", "[mix]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", r["audio_bitrate"],
        "-shortest",
        "-movflags", "+faststart",
        str(video_out),
    ]
    subprocess.run(cmd, check=True)


def _write_summary(label: str, segs: list[dict], final: Path, duration: float, target: int | None) -> None:
    lines = [f"# {label} 高光 — 选段说明\n",
             f"输出文件：`{final.name}`  ·  实际时长 {duration:.1f}s" +
             (f"  ·  目标 {target}s" if target else "") + "\n",
             "## 段落清单\n",
             "| # | 源 | 时间 | 长度 | 慢放 | 选用理由 |",
             "|---|---|---|---|---|---|"]
    for i, s in enumerate(segs, start=1):
        dur = s["end"] - s["start"]
        slow = "✓ 0.7×" if s.get("is_peak_top10") else ""
        reason = s.get("reason", "")
        ts = f"{s['start']:.2f}–{s['end']:.2f}"
        lines.append(f"| {i} | {s['video']} | {ts} | {dur:.2f}s | {slow} | {reason} |")
    summary_path = final.with_suffix(".summary.md")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run(paths: Paths, cfg: dict) -> None:
    plans = load_json(paths.work_dir / "plan.json")
    clip_dir = paths.subdir("clips")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    music_data = None
    music_p = paths.subdir("music") / "onsets.json"
    if music_p.exists():
        d = load_json(music_p)
        if not d.get("skipped"):
            music_data = d

    for label, segs in plans.items():
        log(f"=== {label} ({len(segs)} segments) ===", "render")
        if not segs:
            continue

        brightnesses = []
        for seg in segs:
            src = paths.input_dir / f"{seg['video']}.MP4"
            brightnesses.append(_measure_brightness(src, seg["start"], seg["end"] - seg["start"]))
        target_bright = sorted(brightnesses)[len(brightnesses) // 2]
        log(f"target brightness = {target_bright:.1f}", "render")

        clip_paths: list[Path] = []
        for i, seg in enumerate(segs):
            src = paths.input_dir / f"{seg['video']}.MP4"
            slow = bool(seg.get("is_peak_top10"))
            tag = "slow" if slow else "norm"
            dst = clip_dir / f"{label}_{i:03d}_{seg['video']}_{seg['start']:.2f}-{seg['end']:.2f}_{tag}.mp4"
            if not dst.exists():
                log(f"cut [{tag}] {seg['video']}  {seg['start']:.2f}..{seg['end']:.2f}  br={brightnesses[i]:.0f}",
                    "render")
                _cut(src, seg["start"], seg["end"], dst,
                     slow=slow, target_bright=target_bright, seg_bright=brightnesses[i], cfg=cfg)
            clip_paths.append(dst)

        final = paths.output_dir / f"highlight_{label}.mp4"
        log(f"xfade -> {final}", "render")
        _xfade_chain(clip_paths, final, cfg)

        # Optional music bed (writes to a tmp file, then replaces final)
        if music_data:
            music_path = Path(music_data["audio"])
            if music_path.exists():
                log(f"adding music bed from {music_path.name}", "render")
                final_with_music = final.with_suffix(".mus.mp4")
                _mix_music_bed(final, music_path, final_with_music, cfg)
                final_with_music.replace(final)

        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(final),
        ]).strip()
        duration = float(out)
        target_secs = next((v["seconds"] for v in cfg["durations"]["versions"] if v["label"] == label), None)
        _write_summary(label, segs, final, duration, target_secs)
        log(f"{final.name}: {duration:.1f}s  (summary at {final.with_suffix('.summary.md').name})", "render")
