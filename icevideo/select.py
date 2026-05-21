"""Step 6: fuse signals into per-second scores, find peaks, enforce diversity and
narrative, then snap segment boundaries to silence/scene change points."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from icevideo.config import Paths, discover_videos
from icevideo.utils import (
    cosine_sim, load_json, log, save_json, smooth, video_basename, zscore,
)


def _flatten_keywords(cfg: dict) -> list[str]:
    out: list[str] = []
    for v in cfg["keywords"].values():
        out.extend(v)
    return [w.lower() for w in out]


def _load_signals(paths: Paths, base: str):
    return load_json(paths.subdir("signals") / f"{base}.json")


def _load_words(paths: Paths, base: str) -> list[tuple[float, float, str]]:
    p = paths.subdir("transcripts") / f"{base}.json"
    if not p.exists():
        return []
    data = load_json(p)
    out: list[tuple[float, float, str]] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []) or []:
            out.append((w["start"], w["end"], w["word"]))
    return out


def _per_sec_dialogue(words, n_sec: int, keywords: list[str]):
    density = np.zeros(n_sec, dtype=np.float32)
    bonus = np.zeros(n_sec, dtype=np.float32)
    for ws, we, w in words:
        i0 = max(0, int(math.floor(ws))); i1 = min(n_sec - 1, int(math.floor(we)))
        for i in range(i0, i1 + 1):
            density[i] += 1.0
        wl = w.lower().strip(" .,!?¡¿\"'()")
        if any(k in wl for k in keywords):
            mid = min(n_sec - 1, int((ws + we) / 2))
            bonus[mid] += 1.0
    return smooth(density, 3), bonus


def _load_clip(paths: Paths, base: str, n_sec: int):
    p = paths.subdir("clip_scores") / f"{base}.json"
    emb_p = paths.subdir("clip_embs") / f"{base}.npy"
    if not p.exists():
        return None
    data = load_json(p)
    sims = np.asarray(data["sims"], dtype=np.float32)
    frame_times = data["frame_times"]
    # Either new format (n_positive count) or legacy format (named prompt keys).
    if "n_positive" in data:
        n_pos = data["n_positive"]
        pos_idx = list(range(n_pos))
        neg_idx = list(range(n_pos, sims.shape[1]))
    else:
        legacy = data.get("prompts", [])
        neg_names = {"static", "blurry"}
        neg_idx = [i for i, k in enumerate(legacy) if k in neg_names]
        pos_idx = [i for i in range(sims.shape[1]) if i not in neg_idx]
    per_sec = np.zeros(n_sec, dtype=np.float32)
    counts = np.zeros(n_sec, dtype=np.int32)
    for fi, t in enumerate(frame_times):
        pos = sims[fi, pos_idx].max() if pos_idx else 0.0
        neg = sims[fi, neg_idx].max() if neg_idx else 0.0
        score = pos - 0.5 * neg
        for sec in range(max(0, int(t) - 1), min(n_sec, int(t) + 2)):
            per_sec[sec] += score; counts[sec] += 1
    per_sec = per_sec / np.maximum(counts, 1)
    embs = np.load(emb_p) if emb_p.exists() else None
    return per_sec, embs, frame_times


def _load_faces(paths: Paths, base: str, n_sec: int):
    p = paths.subdir("faces") / f"{base}.json"
    if not p.exists():
        return np.zeros(n_sec, dtype=np.float32), np.zeros(n_sec, dtype=np.float32)
    d = load_json(p)
    if d.get("error") or d.get("skipped"):
        return np.zeros(n_sec, dtype=np.float32), np.zeros(n_sec, dtype=np.float32)
    face = np.asarray(d.get("per_sec_face_area", []), dtype=np.float32)
    prot = np.asarray(d.get("per_sec_protagonist", []), dtype=np.float32)
    def pad(a):
        if len(a) >= n_sec: return a[:n_sec]
        return np.concatenate([a, np.zeros(n_sec - len(a), dtype=np.float32)])
    return pad(face), pad(prot)


def _load_diarize(paths: Paths, base: str, n_sec: int):
    """Return (protagonist_speech_per_sec, total_speech_per_sec)."""
    p = paths.subdir("diarize") / f"{base}.json"
    if not p.exists():
        return np.zeros(n_sec, dtype=np.float32), np.zeros(n_sec, dtype=np.float32)
    d = load_json(p)
    turns = d.get("turns", []) or []
    if not turns:
        return np.zeros(n_sec, dtype=np.float32), np.zeros(n_sec, dtype=np.float32)
    # Most-frequent speaker = protagonist
    from collections import Counter
    c = Counter(t["speaker"] for t in turns)
    protagonist = c.most_common(1)[0][0] if c else None
    prot = np.zeros(n_sec, dtype=np.float32)
    total = np.zeros(n_sec, dtype=np.float32)
    for t in turns:
        i0 = max(0, int(t["start"])); i1 = min(n_sec - 1, int(t["end"]))
        for i in range(i0, i1 + 1):
            total[i] = 1.0
            if t["speaker"] == protagonist:
                prot[i] = 1.0
    return prot, total


def _telemetry_bonus(paths: Paths, base: str, n_sec: int) -> np.ndarray:
    """Convert telemetry events into a per-second bonus (golden hour + GPS jump anchors)."""
    p = paths.subdir("telemetry") / f"{base}.json"
    bonus = np.zeros(n_sec, dtype=np.float32)
    if not p.exists():
        return bonus
    d = load_json(p)
    if d.get("error"):
        return bonus
    # Apply golden_hour bonus uniformly (small constant per sec)
    bonus += float(d.get("golden_hour", 0.0)) * 0.5
    for ev in d.get("events", []) or []:
        if ev.get("type") == "gps_jump":
            i = int(round(ev["t"]))
            if 0 <= i < n_sec:
                # GPS jump = "we changed places", boost a 3-second window around it
                for k in range(max(0, i - 1), min(n_sec, i + 2)):
                    bonus[k] += 0.5
    return bonus


def _compute_per_video(paths: Paths, cfg: dict):
    keywords = _flatten_keywords(cfg)
    w = cfg["scoring"]
    out: dict[str, dict] = {}
    videos = discover_videos(paths)
    for video in videos:
        base = video_basename(video)
        sig = _load_signals(paths, base)
        n_sec = sig["seconds"]
        scene = np.asarray(sig["scene"], dtype=np.float32)
        flow = np.asarray(sig["flow"], dtype=np.float32)
        rms = np.asarray(sig["audio_rms_db"], dtype=np.float32)
        voice = np.asarray(sig["voiceness"], dtype=np.float32)
        laugh = np.asarray(sig["laughter"], dtype=np.float32)

        words = _load_words(paths, base)
        density, kw_bonus = _per_sec_dialogue(words, n_sec, keywords)

        clip_data = _load_clip(paths, base, n_sec)
        if clip_data is not None:
            clip_score, embs, frame_times = clip_data
        else:
            clip_score = np.zeros(n_sec, dtype=np.float32)
            embs = None; frame_times = []

        face_area, prot_face = _load_faces(paths, base, n_sec)
        prot_speech, _total_speech = _load_diarize(paths, base, n_sec)
        telem_bonus = _telemetry_bonus(paths, base, n_sec)

        gate = 0.3 + 0.7 * voice
        gated_rms = smooth(rms, 3) * gate

        score = (
            w["motion"] * zscore(smooth(scene, 3))
            + w["flow"] * zscore(smooth(flow, 3))
            + w["audio"] * zscore(gated_rms)
            + w["dialogue"] * zscore(smooth(density, 3))
            + w["laughter"] * smooth(laugh, 3)
            + w["keyword"] * kw_bonus
            + w["clip"] * zscore(smooth(clip_score, 3))
            + w.get("faces", 0.6) * smooth(prot_face + 0.4 * face_area, 3)
            + w.get("protagonist_speech", 0.5) * smooth(prot_speech, 3)
            + w.get("telemetry", 0.4) * smooth(telem_bonus, 3)
        )
        if n_sec >= 2:
            score[0] -= 0.5; score[-1] -= 0.5

        out[base] = {
            "score": score, "n_sec": n_sec,
            "embs": embs, "frame_times": frame_times,
            "voiceness": voice,
            "signal_breakdown": {
                "scene_max": float(scene.max()),
                "flow_max": float(flow.max()),
                "voiceness_ratio": float((voice > 0.4).mean()),
                "laughter_max": float(laugh.max()),
                "clip_max": float(clip_score.max()) if clip_data is not None else 0.0,
                "face_max": float(face_area.max()),
                "protagonist_ratio": float((prot_face > 0).mean()),
                "telemetry_max": float(telem_bonus.max()),
            },
        }
    return out, [video_basename(v) for v in videos]


def _find_peak_segments(score: np.ndarray, min_len: int, max_len: int, pad: int):
    n = len(score)
    used = np.zeros(n, bool)
    segs = []
    order = np.argsort(-score)
    for idx in order:
        if used[idx]:
            continue
        if score[idx] < -1.0:
            break
        s = e = int(idx)
        while (e - s + 1) < max_len:
            grew = False
            if s > 0 and not used[s - 1] and score[s - 1] > -0.4:
                s -= 1; grew = True
            if e < n - 1 and not used[e + 1] and score[e + 1] > -0.4:
                e += 1; grew = True
            if not grew:
                break
        while (e - s + 1) < min_len:
            if s > 0 and not used[s - 1]:
                s -= 1
            elif e < n - 1 and not used[e + 1]:
                e += 1
            else:
                break
        s = max(0, s - pad); e = min(n - 1, e + pad)
        if used[s:e + 1].any():
            continue
        used[s:e + 1] = True
        segs.append({
            "start": int(s), "end": int(e) + 1,
            "score": float(np.sum(score[s:e + 1])),
            "peak": int(idx),
        })
    return segs


def _emb_at_time(embs, frame_times, t: float):
    if embs is None or not frame_times:
        return None
    return embs[int(np.argmin([abs(t - ft) for ft in frame_times]))]


def _pick(per_video, video_order, budget_sec: int, scfg: dict, all_segs: dict):
    sim_thr = scfg["diversity_threshold"]
    chosen: list[dict] = []
    total = 0

    def add(seg, thr: float) -> bool:
        nonlocal total
        if any(cosine_sim(seg.get("emb"), c.get("emb")) > thr for c in chosen):
            return False
        d = seg["end"] - seg["start"]
        if total + d > budget_sec:
            remaining = budget_sec - total
            if remaining < 3:
                return False
            center = (seg["start"] + seg["end"]) / 2
            half = remaining / 2
            new_s = max(seg["start"], int(center - half))
            new_e = min(seg["end"], new_s + remaining)
            if new_e - new_s < 3:
                return False
            seg = {**seg, "start": int(new_s), "end": int(new_e)}
            d = seg["end"] - seg["start"]
        chosen.append(seg); total += d
        return True

    # 1) Per-video quota — force-include best from a few early-listed videos
    max_quota = max(1, budget_sec // 20)
    quota_pool = [max(all_segs[v], key=lambda s: s["score"]) for v in video_order if all_segs.get(v)]
    quota_pool.sort(key=lambda s: -s["score"])
    for seg in quota_pool[:max_quota]:
        add(seg, sim_thr)

    # 2) Greedy by score
    remaining = []
    for segs in all_segs.values():
        remaining.extend(s for s in segs if s not in chosen)
    remaining.sort(key=lambda s: -s["score"])
    for seg in remaining:
        if total >= budget_sec:
            break
        add(seg, sim_thr)

    # 3) Budget fill — relax diversity gradually so long versions reach target
    for relax in (0.95, 0.98, 1.01):
        if total >= budget_sec * 0.95:
            break
        for seg in remaining:
            if total >= budget_sec * 0.98:
                break
            if seg in chosen:
                continue
            add(seg, relax)

    # 4) Chronological reorder
    chosen.sort(key=lambda s: (video_order.index(s["video"]), s["start"]))

    # 5) Narrative open/close
    if budget_sec >= 30 and video_order:
        early = set(video_order[:max(1, len(video_order) // 5)])
        late = set(video_order[-max(1, len(video_order) // 5):])
        if not any(s["video"] in early for s in chosen):
            for v in video_order[:3]:
                cands = all_segs.get(v, [])
                if cands:
                    pick = max(cands, key=lambda s: s["score"])
                    if pick not in chosen:
                        weak = min(chosen, key=lambda s: s["score"], default=None)
                        if weak and weak["video"] not in early and weak["video"] not in late:
                            total -= weak["end"] - weak["start"]; chosen.remove(weak)
                        if add(pick, 1.01):
                            break
        if not any(s["video"] in late for s in chosen):
            for v in reversed(video_order[-3:]):
                cands = all_segs.get(v, [])
                if cands:
                    pick = max(cands, key=lambda s: s["score"])
                    if pick not in chosen:
                        weak = min(chosen, key=lambda s: s["score"], default=None)
                        if weak and weak["video"] not in early and weak["video"] not in late:
                            total -= weak["end"] - weak["start"]; chosen.remove(weak)
                        if add(pick, 1.01):
                            break
        chosen.sort(key=lambda s: (video_order.index(s["video"]), s["start"]))
    return chosen, total


def _snap_boundary(t: float, silence, scenes, window: float) -> float:
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
    scene_cands = [c for c in cands if c[0] == "scene" and c[2] <= 0.4]
    if scene_cands:
        return min(scene_cands, key=lambda c: c[2])[1]
    return min(cands, key=lambda c: c[2])[1]


def _refine_segment(seg: dict, silence, scenes, window: float, tol: float) -> tuple[float, float]:
    s0, e0 = float(seg["start"]), float(seg["end"])
    orig_dur = e0 - s0
    s_new = _snap_boundary(s0, silence, scenes, window)
    e_new = _snap_boundary(e0, silence, scenes, window)
    new_dur = e_new - s_new
    if new_dur < orig_dur * (1 - tol) or new_dur > orig_dur * (1 + tol):
        s_new, e_new = s0, e0
    s_q = round(s_new / 0.04) * 0.04
    e_q = round(e_new / 0.04) * 0.04
    return s_q, e_q


def _build_reason(seg: dict, info: dict, breakdown: dict) -> str:
    parts = []
    s, e = seg["start"], seg["end"]
    score = info["score"]
    seg_max = float(score[s:e].max()) if e > s else 0.0
    if breakdown["clip_max"] > 0.3 and seg_max > 0:
        parts.append(f"strong CLIP visual ({breakdown['clip_max']:.2f})")
    if breakdown["laughter_max"] > 0.5:
        parts.append("laughter detected")
    if breakdown["protagonist_ratio"] > 0.3:
        parts.append("protagonist on-screen")
    if breakdown["scene_max"] > 0.4:
        parts.append("dynamic motion")
    if seg.get("score", 0) > 30:
        parts.append("top decile across all videos")
    if not parts:
        parts.append("balanced score peak")
    return "; ".join(parts)


def run(paths: Paths, cfg: dict) -> None:
    """Produce work/plan.json — a *candidate* plan, not a commitment.
    The AI is expected to read it, edit it (or ignore it) and then call cut/concat.
    """
    sel = cfg["selection"]
    bcfg = cfg["boundaries"]

    per_video, video_order = _compute_per_video(paths, cfg)
    all_segs: dict[str, list[dict]] = {}
    for base, info in per_video.items():
        segs = _find_peak_segments(info["score"], sel["min_seg"], sel["max_seg"], sel["peak_pad"])
        for s in segs:
            mid = (s["start"] + s["end"]) / 2
            s["video"] = base
            s["emb"] = _emb_at_time(info["embs"], info["frame_times"], mid)
        all_segs[base] = segs
        log(f"{base}: peak_score={info['score'].max():.2f}  segs={len(segs)}", "select")

    plans: dict[str, list[dict]] = {}
    for spec in cfg["durations"]["versions"]:
        label = spec["label"]; budget = int(spec["seconds"])
        chosen, total = _pick(per_video, video_order, budget, sel, all_segs)
        log(f"{label} plan: {total}s / target {budget}s ({len(chosen)} segs)", "select")
        refined = []
        for seg in chosen:
            bjson = paths.subdir("boundaries") / f"{seg['video']}.json"
            silence = scenes = []
            if bjson.exists():
                b = load_json(bjson)
                silence = b["silence"]; scenes = b["scenes"]
            s_r, e_r = _refine_segment(seg, silence, scenes,
                                       bcfg["snap_window"], bcfg["duration_tolerance"])
            info = per_video[seg["video"]]
            refined.append({
                "video": seg["video"], "start": s_r, "end": e_r,
                "score": seg["score"], "is_peak_top10": False,
                "reason": _build_reason(seg, info, info["signal_breakdown"]),
            })
        scores = sorted([r["score"] for r in refined], reverse=True)
        if scores:
            cutoff_idx = max(0, len(scores) * sel["peak_top_pct"] // 100 - 1)
            cutoff = scores[cutoff_idx]
            for r in refined:
                r["is_peak_top10"] = r["score"] >= cutoff
        plans[label] = refined

    save_json(paths.work_dir / "plan.json", plans)
    log(f"wrote {paths.work_dir / 'plan.json'}", "select")
