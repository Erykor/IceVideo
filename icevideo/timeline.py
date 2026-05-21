"""Print a per-second readable timeline of one video.

Pulls together everything in work/: signals, transcripts, clip_scores, faces, diarize.
Output is a markdown table or CSV — the AI reads it the same way a human would scan a
timeline in a video editor.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from icevideo.utils import load_json, video_basename


def _try_load(p: Path):
    if not p.exists():
        return None
    try:
        return load_json(p)
    except Exception:
        return None


def _try_load_npy(p: Path):
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception:
        return None


def _per_sec_words(transcript: dict | None, n_sec: int) -> list[str]:
    out = [""] * n_sec
    if not transcript:
        return out
    for seg in transcript.get("segments", []) or []:
        text = seg.get("text", "").strip()
        if not text:
            continue
        s = max(0, int(seg.get("start", 0)))
        e = min(n_sec - 1, int(seg.get("end", n_sec)))
        # put the whole segment text on its mid-second; if multi-word the AI can scan
        mid = max(s, min(e, (s + e) // 2))
        if out[mid]:
            out[mid] += " " + text
        else:
            out[mid] = text
    return out


def _per_sec_speakers(diarize: dict | None, n_sec: int) -> list[str]:
    out = [""] * n_sec
    if not diarize:
        return out
    for t in diarize.get("turns", []) or []:
        s = max(0, int(t["start"])); e = min(n_sec - 1, int(t["end"]))
        spk = t["speaker"].replace("SPEAKER_", "S")
        for i in range(s, e + 1):
            out[i] = spk
    return out


def _per_sec_clip(clip_scores: dict | None, n_sec: int) -> tuple[list[float], list[str]]:
    score = [0.0] * n_sec
    label = [""] * n_sec
    if not clip_scores:
        return score, label
    sims = np.asarray(clip_scores["sims"], dtype=np.float32)
    times = clip_scores["frame_times"]
    prompts = clip_scores.get("prompts") or clip_scores.get("keys") or []
    if "n_positive" in clip_scores:
        n_pos = clip_scores["n_positive"]
        pos_idx = list(range(n_pos))
    else:
        neg_names = {"static", "blurry"}
        pos_idx = [i for i, k in enumerate(prompts) if k not in neg_names]
    for fi, t in enumerate(times):
        if not pos_idx:
            break
        best = int(np.argmax(sims[fi, pos_idx]))
        s = max(0, int(t) - 1); e = min(n_sec - 1, int(t) + 1)
        for sec in range(s, e + 1):
            if sims[fi, pos_idx[best]] > score[sec]:
                score[sec] = float(sims[fi, pos_idx[best]])
                # try to give a short label
                lbl = str(prompts[pos_idx[best]])[:24]
                label[sec] = lbl
    return score, label


def build_timeline(video: Path, work_dir: Path) -> tuple[dict, list[dict]]:
    base = video_basename(video)
    sig = _try_load(work_dir / "signals" / f"{base}.json")
    transcript = _try_load(work_dir / "transcripts" / f"{base}.json")
    clip_scores = _try_load(work_dir / "clip_scores" / f"{base}.json")
    faces_data = _try_load(work_dir / "faces" / f"{base}.json")
    diarize = _try_load(work_dir / "diarize" / f"{base}.json")

    if not sig:
        # fall back to bare duration probe
        from icevideo.utils import probe_duration
        n_sec = int(probe_duration(video))
        sig = {"seconds": n_sec, "scene": [0] * n_sec, "flow": [0] * n_sec,
               "audio_rms_db": [-80] * n_sec, "voiceness": [0] * n_sec, "laughter": [0] * n_sec}

    n_sec = sig["seconds"]
    scene = sig["scene"]; flow = sig["flow"]; rms = sig["audio_rms_db"]
    voice = sig["voiceness"]; laugh = sig["laughter"]
    words = _per_sec_words(transcript, n_sec)
    speakers = _per_sec_speakers(diarize, n_sec)
    clip_score, clip_label = _per_sec_clip(clip_scores, n_sec)
    face_area = (faces_data or {}).get("per_sec_face_area") or [0.0] * n_sec
    if len(face_area) < n_sec:
        face_area = list(face_area) + [0.0] * (n_sec - len(face_area))

    rows = []
    for i in range(n_sec):
        rows.append({
            "sec": i,
            "scene": round(scene[i], 2),
            "flow": round(flow[i], 1),
            "dB": round(rms[i], 1),
            "voice": round(voice[i], 2),
            "laugh": round(laugh[i], 2),
            "clip": round(clip_score[i], 2),
            "clip_label": clip_label[i],
            "face": round(face_area[i], 3),
            "speaker": speakers[i],
            "text": words[i],
        })

    meta = {
        "video": base,
        "duration_s": n_sec,
        "has_signals": bool(sig),
        "has_transcript": bool(transcript),
        "has_clip": bool(clip_scores),
        "has_faces": bool(faces_data),
        "has_diarize": bool(diarize),
    }
    return meta, rows


def render_markdown(meta: dict, rows: list[dict], *, only_active: bool = False) -> str:
    """Markdown table. With --only-active, hide rows that are visually/aurally flat."""
    lines = [
        f"# Timeline: {meta['video']} ({meta['duration_s']}s)\n",
        "| sec | scene | flow | dB  | voice | laugh | clip | clip_label | face | spk | text |",
        "|----:|------:|-----:|----:|------:|------:|-----:|:-----------|-----:|:----|:-----|",
    ]
    for r in rows:
        active = (r["scene"] > 0.1 or r["laugh"] > 0.2 or r["voice"] > 0.4
                  or r["clip"] > 0.25 or r["face"] > 0 or r["text"])
        if only_active and not active:
            continue
        lines.append(
            f"| {r['sec']:>3} | {r['scene']:>5.2f} | {r['flow']:>4.1f} | {r['dB']:>5.1f} | "
            f"{r['voice']:>5.2f} | {r['laugh']:>5.2f} | {r['clip']:>4.2f} | "
            f"{r['clip_label']:<24} | {r['face']:>4.3f} | {r['speaker']:<3} | {r['text']} |"
        )
    return "\n".join(lines)


def render_csv(meta: dict, rows: list[dict]) -> str:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [
        "sec", "scene", "flow", "dB", "voice", "laugh", "clip", "clip_label",
        "face", "speaker", "text",
    ])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def cli(video: Path, work_dir: Path, *, fmt: str = "md", only_active: bool = False,
        out: Path | None = None) -> None:
    meta, rows = build_timeline(video, work_dir)
    text = render_csv(meta, rows) if fmt == "csv" else render_markdown(meta, rows, only_active=only_active)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")
