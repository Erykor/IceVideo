"""Inspect a video: duration, codec, resolution, fps, audio.

Used by the AI to orient itself before deciding what to do. Outputs JSON-friendly text
for easy parsing or piping into other tools.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe(video: Path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(video),
    ])
    data = json.loads(raw)
    fmt = data.get("format", {})
    vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    astream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})

    fps = None
    if rate := vstream.get("r_frame_rate"):
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else None
        except Exception:
            fps = None

    return {
        "path": str(video),
        "duration_s": float(fmt.get("duration", 0)),
        "size_mb": int(fmt.get("size", 0)) / (1024 * 1024),
        "bitrate_kbps": int(int(fmt.get("bit_rate", 0)) / 1000) if fmt.get("bit_rate") else None,
        "video": {
            "codec": vstream.get("codec_name"),
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "fps": round(fps, 2) if fps else None,
            "pix_fmt": vstream.get("pix_fmt"),
        },
        "audio": {
            "codec": astream.get("codec_name"),
            "channels": astream.get("channels"),
            "sample_rate": int(astream.get("sample_rate", 0)) or None,
        } if astream else None,
        "creation_time": fmt.get("tags", {}).get("creation_time"),
    }


def cli(video_paths: list[Path], as_json: bool = False) -> None:
    results = []
    for v in video_paths:
        try:
            results.append(probe(v))
        except Exception as e:
            results.append({"path": str(v), "error": str(e)})

    if as_json:
        print(json.dumps(results, indent=2))
        return

    # Compact table
    print(f"{'file':<30} {'dur':>6} {'res':<11} {'fps':>5} {'codec':<6} {'audio':<10}")
    for r in results:
        if "error" in r:
            print(f"{Path(r['path']).name:<30}  ERROR: {r['error']}")
            continue
        v = r["video"]; a = r["audio"] or {}
        res = f"{v['width']}x{v['height']}" if v["width"] else "-"
        print(f"{Path(r['path']).name:<30} {r['duration_s']:>5.1f}s {res:<11} "
              f"{v['fps'] or 0:>5.1f} {v['codec'] or '-':<6} "
              f"{(a.get('codec') or '-')+'/'+str(a.get('channels') or '-')+'ch':<10}")
