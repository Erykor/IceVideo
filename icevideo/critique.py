"""Step 8: self-evaluate the rendered video.

If an Anthropic / Google API key is in the env, we send sampled frames + captions to
a VLM for rubric-scoring. Otherwise we fall back to a deterministic heuristic that
checks the same dimensions using local signals only.

Output `work/critique/<label>.json` with:
  {
    "engine": "vlm" | "heuristic",
    "scores": { pacing, hook, narrative, variety, ending, overall },
    "suggestions": [{target, issue, fix}, ...],
    "iterate": bool
  }
The `iterate` flag tells the auto runner whether to try again. Capped at 2 iterations
upstream.
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from icevideo.config import Paths
from icevideo.utils import load_json, log, save_json


# ----------------------------------------------------------------------------
# Heuristic critique (always available)
# ----------------------------------------------------------------------------

def heuristic_critique(label: str, plan: list[dict], paths: Paths, cfg: dict) -> dict:
    target = next((v["seconds"] for v in cfg["durations"]["versions"] if v["label"] == label), None)
    n = len(plan)
    if n == 0:
        return {"engine": "heuristic", "scores": {"overall": 0}, "suggestions": [], "iterate": False}

    # Pacing: penalize segments < 2s or > 15s
    durs = [s["end"] - s["start"] for s in plan]
    too_short = sum(1 for d in durs if d < 2.0)
    too_long = sum(1 for d in durs if d > 15.0)
    pacing_penalty = (too_short + too_long) / n
    pacing = max(0, 10 - pacing_penalty * 15)

    # Hook: is first segment within top 20% by score?
    scores = sorted([s.get("score", 0) for s in plan], reverse=True)
    cutoff = scores[max(0, n // 5 - 1)] if scores else 0
    hook = 10.0 if plan[0].get("score", 0) >= cutoff else 5.0

    # Variety: how many distinct source videos in use vs available
    used = {s["video"] for s in plan}
    total = len(list(paths.subdir("signals").glob("*.json")))
    if total == 0:
        # No signals data — base variety only on internal repetition, not coverage
        variety = 8.0
    else:
        variety = 10.0 * min(1.0, len(used) / total)
    # ... but punish if >50% of segments come from one video
    counts = Counter(s["video"] for s in plan)
    dominance = max(counts.values()) / n
    if dominance > 0.5:
        variety *= (1 - (dominance - 0.5))

    # Narrative: does it use both early and late videos?
    # Use video name string ordering as a proxy for chronology.
    videos_sorted = sorted(used)
    if total >= 5 and videos_sorted:
        first_q = sorted(s["video"] for s in plan)[: max(1, n // 5)]
        last_q = sorted(s["video"] for s in plan)[-max(1, n // 5):]
        has_early = any(v == videos_sorted[0] for v in first_q)
        has_late = any(v == videos_sorted[-1] for v in last_q)
        narrative = 8.0 + (1.0 if has_early else 0) + (1.0 if has_late else 0)
    else:
        narrative = 7.0

    # Ending: last segment score in top 20%?
    ending = 10.0 if plan[-1].get("score", 0) >= cutoff else 6.0

    # Adjacent similarity (visual repetition)
    rep = 0
    for i in range(1, n):
        if plan[i]["video"] == plan[i - 1]["video"]:
            rep += 1
    variety_penalty = rep / max(n - 1, 1)
    variety = variety * (1 - variety_penalty * 0.5)

    # Duration accuracy
    total_s = sum(durs)
    dur_drift = abs(total_s - target) / target if target else 0
    duration_score = max(0, 10 - dur_drift * 100)

    overall = float(np.mean([pacing, hook, narrative, variety, ending, duration_score]))

    suggestions = []
    if pacing < 6:
        suggestions.append({"target": "selection", "issue": "many segments outside [2s, 15s]",
                            "fix": "increase selection.min_seg or selection.max_seg"})
    if hook < 7:
        suggestions.append({"target": "opening", "issue": "first segment is below top 20% by score",
                            "fix": "enforce_hook=true: pick highest-score segment first"})
    if variety < 6:
        suggestions.append({"target": "selection", "issue": f"one video dominates ({dominance:.0%})",
                            "fix": "lower selection.per_video_max or tighten diversity_threshold"})
    if narrative < 8:
        suggestions.append({"target": "narrative", "issue": "missing early or late source videos",
                            "fix": "enforce_opener/closer=true"})

    return {
        "engine": "heuristic",
        "scores": {
            "pacing": round(pacing, 1),
            "hook": round(hook, 1),
            "narrative": round(narrative, 1),
            "variety": round(variety, 1),
            "ending": round(ending, 1),
            "duration": round(duration_score, 1),
            "overall": round(overall, 1),
        },
        "suggestions": suggestions,
        "iterate": overall < float(cfg.get("critique", {}).get("iterate_threshold", 7.0)),
    }


# ----------------------------------------------------------------------------
# VLM critique (used when ANTHROPIC_API_KEY is set)
# ----------------------------------------------------------------------------

def _extract_thumbnails(video: Path, n_thumbs: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]).strip())
    for i in range(n_thumbs):
        t = dur * (i + 0.5) / n_thumbs
        p = out_dir / f"thumb_{i:02d}.jpg"
        subprocess.run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
            "-vf", "scale=640:-1", str(p),
        ], check=True)
        out.append(p)
    return out


def vlm_critique(label: str, video: Path, paths: Paths, cfg: dict) -> dict | None:
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    ccfg = cfg.get("critique", {})
    n_thumbs = int(ccfg.get("n_thumbnails", 10))
    thumb_dir = paths.subdir("critique") / f"{label}_thumbs"
    thumbs = _extract_thumbnails(video, n_thumbs, thumb_dir)

    import base64
    content = [{"type": "text", "text": (
        f"You are a video editor reviewer. The clip is a {label} highlight reel.\n"
        "Below are evenly-spaced thumbnails. Score on the following rubric (0-10):\n"
        "  pacing, hook (first 3s), narrative arc, variety, ending impact, overall\n"
        "Then list at most 4 actionable suggestions. Return strict JSON:\n"
        "  {\"scores\": {pacing, hook, narrative, variety, ending, overall},\n"
        "   \"suggestions\": [{\"target\": string, \"issue\": string, \"fix\": string}]}"
    )}]
    for p in thumbs:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                        "data": base64.b64encode(p.read_bytes()).decode("ascii")},
        })

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=ccfg.get("vlm_model", "claude-haiku-4-5-20251001"),
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text if resp.content else "{}"
    try:
        import json, re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}
    scores = data.get("scores", {})
    overall = float(scores.get("overall", 0))
    return {
        "engine": "vlm",
        "scores": {k: float(v) for k, v in scores.items()},
        "suggestions": data.get("suggestions", []),
        "iterate": overall < float(ccfg.get("iterate_threshold", 7.0)),
        "raw": text,
    }


def critique_one(label: str, plan: list[dict], paths: Paths, cfg: dict) -> dict:
    final = paths.output_dir / f"highlight_{label}.mp4"
    result = None
    if final.exists() and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            result = vlm_critique(label, final, paths, cfg)
        except Exception as e:
            log(f"vlm critique failed — {e}; falling back to heuristic", "critique")
    if not result:
        result = heuristic_critique(label, plan, paths, cfg)
    return result


def run(paths: Paths, cfg: dict) -> None:
    plan_path = paths.work_dir / "plan.json"
    if not plan_path.exists():
        log("no plan.json — run select first", "critique"); return
    plans = load_json(plan_path)
    out = {}
    for label, segs in plans.items():
        r = critique_one(label, segs, paths, cfg)
        out[label] = r
        log(f"{label}: engine={r['engine']}  overall={r['scores'].get('overall', '?')}  iterate={r['iterate']}",
            "critique")
        for s in r["suggestions"]:
            log(f"  - {s.get('target')}: {s.get('issue')}  → {s.get('fix')}", "critique")
    save_json(paths.work_dir / "critique.json", out)
    log(f"wrote {paths.work_dir / 'critique.json'}", "critique")
