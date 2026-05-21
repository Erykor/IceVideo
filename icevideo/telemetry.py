"""Step 4.5: read camera-side metadata where available.

Two tiers:
  1. ffprobe `format=creation_time` + EXIF DateTimeOriginal → start_time
     (works for almost any camera-recorded file)
  2. GoPro GPMF telemetry → GPS/speed/altitude per second
     (only if `gopro-telemetry-parser` or similar is installed; we try a pure-Python
     fallback using `subprocess` to call exiftool if available)

This step is best-effort. Missing metadata is normal and downstream code never relies
on telemetry being present.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
import subprocess
from pathlib import Path

from icevideo.config import Paths, discover_videos
from icevideo.utils import log, save_json, video_basename


def _ffprobe_metadata(video: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(video),
    ])
    return json.loads(out)


def _parse_iso_or_none(s: str | None) -> str | None:
    if not s:
        return None
    try:
        dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return s
    except Exception:
        return None


def _golden_hour_score(start_iso: str | None) -> float:
    """Crude prior: how 'golden' is the time of day at recording.
    Returns 0..1 where 1 ≈ sunset/sunrise band, 0 ≈ midday or night.
    Uses local hour from the timestamp; no lat/lon-aware sun calc.
    """
    if not start_iso:
        return 0.0
    try:
        t = dt.datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    h = t.hour + t.minute / 60.0
    # Two peaks: 06:30-08:30 and 17:00-19:30
    def bump(center: float, width: float) -> float:
        return max(0.0, 1 - ((h - center) / width) ** 2)
    return max(bump(7.5, 1.5), bump(18.25, 2.0))


def _exiftool_gpmf(video: Path) -> list[dict] | None:
    if shutil.which("exiftool") is None:
        return None
    try:
        out = subprocess.check_output([
            "exiftool", "-ee", "-G3", "-json", "-n",
            "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-GPSDateTime", "-GPSSpeed",
            str(video),
        ])
        data = json.loads(out)
        if not data:
            return None
        # exiftool gives a single object per file with GPMF samples flattened; we extract a list-shape.
        d = data[0]
        samples: list[dict] = []
        # GPMF entries come back as e.g. "Doc1:GPSLatitude" / "Doc2:GPSLatitude" etc.
        bydoc: dict[str, dict] = {}
        for k, v in d.items():
            if ":" not in k:
                continue
            doc, name = k.split(":", 1)
            if name in {"GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSDateTime", "GPSSpeed"}:
                bydoc.setdefault(doc, {})[name] = v
        for doc, fields in bydoc.items():
            if "GPSLatitude" in fields and "GPSLongitude" in fields:
                samples.append({
                    "lat": fields.get("GPSLatitude"),
                    "lon": fields.get("GPSLongitude"),
                    "alt_m": fields.get("GPSAltitude"),
                    "speed_mps": fields.get("GPSSpeed"),
                    "t_iso": fields.get("GPSDateTime"),
                })
        return samples or None
    except Exception:
        return None


def _detect_gps_jumps(samples: list[dict], threshold_km: float = 0.5) -> list[float]:
    """Return relative second-offsets (from sample[0]) where GPS jumped > threshold."""
    if not samples or len(samples) < 2:
        return []
    # parse iso → datetime
    times: list[dt.datetime] = []
    for s in samples:
        try:
            times.append(dt.datetime.fromisoformat(s["t_iso"].replace("Z", "+00:00")))
        except Exception:
            return []
    t0 = times[0]
    events: list[float] = []
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if a.get("lat") is None or b.get("lat") is None:
            continue
        d = _haversine_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
        if d > threshold_km:
            events.append((times[i] - t0).total_seconds())
    return events


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def telemetry_for(video: Path) -> dict:
    meta = _ffprobe_metadata(video)
    fmt = meta.get("format", {})
    tags = fmt.get("tags", {})
    start = _parse_iso_or_none(tags.get("creation_time"))

    samples = _exiftool_gpmf(video) or []
    events: list[dict] = []
    if samples:
        for jump_t in _detect_gps_jumps(samples):
            events.append({"t": jump_t, "type": "gps_jump"})
        # speed peaks
        speeds = [(i, float(s.get("speed_mps") or 0)) for i, s in enumerate(samples)]
        if speeds:
            mx_i, mx_v = max(speeds, key=lambda kv: kv[1])
            if mx_v > 1.0:  # > 3.6 km/h: a meaningful movement
                # rough time = mx_i / len(samples) * total_dur, but we don't know dur here;
                # treat sample index as monotonic time within recording duration
                events.append({"t_sample_idx": mx_i, "type": "speed_peak", "speed_mps": mx_v})

    return {
        "video": video.stem,
        "start_iso": start,
        "golden_hour": _golden_hour_score(start),
        "n_gps_samples": len(samples),
        "events": events,
    }


def run(paths: Paths, cfg: dict) -> None:
    out_dir = paths.subdir("telemetry")
    for v in discover_videos(paths):
        base = video_basename(v)
        out_json = out_dir / f"{base}.json"
        if out_json.exists():
            log(f"{base}: skip", "telemetry"); continue
        try:
            t = telemetry_for(v)
            save_json(out_json, t)
            log(f"{base}: golden={t['golden_hour']:.2f}  gps_samples={t['n_gps_samples']}  events={len(t['events'])}",
                "telemetry")
        except Exception as e:
            save_json(out_json, {"video": base, "error": str(e)})
            log(f"{base}: error — {e}", "telemetry")
