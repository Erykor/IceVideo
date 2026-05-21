"""Step 6.5: lint the plan before rendering. Catches choppy/repetitive/cut-mid-word issues."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from icevideo.config import Paths
from icevideo.utils import cosine_sim, load_json, log


@dataclass
class LintFinding:
    level: str           # "info" | "warn" | "error"
    rule: str
    message: str
    plan_index: int | None = None


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)

    def add(self, **kw):
        self.findings.append(LintFinding(**kw))

    @property
    def errors(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warns(self) -> list[LintFinding]:
        return [f for f in self.findings if f.level == "warn"]


def _load_words_for(paths: Paths, base: str) -> list[tuple[float, float]]:
    p = paths.subdir("transcripts") / f"{base}.json"
    if not p.exists():
        return []
    d = load_json(p)
    out = []
    for seg in d.get("segments", []):
        for w in seg.get("words", []) or []:
            out.append((w["start"], w["end"]))
    return out


def _load_emb(paths: Paths, base: str, t: float):
    p = paths.subdir("clip_embs") / f"{base}.npy"
    cs = paths.subdir("clip_scores") / f"{base}.json"
    if not p.exists() or not cs.exists():
        return None
    embs = np.load(p)
    times = load_json(cs)["frame_times"]
    if not times:
        return None
    return embs[int(np.argmin([abs(t - ft) for ft in times]))]


def check_plan(plans: dict[str, list[dict]], paths: Paths, cfg: dict) -> LintReport:
    rules = cfg.get("lint", {})
    min_seg = rules.get("min_seg", 2.0)
    max_seg = rules.get("max_seg", 15.0)
    max_same_source = rules.get("max_consecutive_same_source", 3)
    adj_sim_max = rules.get("adjacent_sim_max", 0.93)
    duration_tol = rules.get("total_duration_tolerance", 0.05)

    report = LintReport()
    for label, segs in plans.items():
        # total duration check (use raw source durations; xfade overlap is accounted for elsewhere)
        target = next((v["seconds"] for v in cfg["durations"]["versions"] if v["label"] == label), None)
        if target:
            total = sum(s["end"] - s["start"] for s in segs)
            drift = (total - target) / target
            level = "warn" if abs(drift) > duration_tol else "info"
            report.add(level=level, rule="total_duration",
                       message=f"{label}: {total:.1f}s / target {target}s ({drift*100:+.1f}%)")

        for i, seg in enumerate(segs):
            dur = seg["end"] - seg["start"]
            base = seg["video"]
            if dur < min_seg:
                report.add(level="warn", rule="short_clip", plan_index=i,
                           message=f"{label}[{i}]: {base} duration {dur:.2f}s < {min_seg}s")
            if dur > max_seg:
                report.add(level="warn", rule="long_clip", plan_index=i,
                           message=f"{label}[{i}]: {base} duration {dur:.2f}s > {max_seg}s")

            # cut mid-word check
            for ws, we in _load_words_for(paths, base):
                if ws + 0.05 < seg["start"] < we - 0.05:
                    report.add(level="error", rule="cut_mid_word_start", plan_index=i,
                               message=f"{label}[{i}]: {base} start {seg['start']:.2f} inside word ({ws:.2f}..{we:.2f})")
                if ws + 0.05 < seg["end"] < we - 0.05:
                    report.add(level="error", rule="cut_mid_word_end", plan_index=i,
                               message=f"{label}[{i}]: {base} end {seg['end']:.2f} inside word ({ws:.2f}..{we:.2f})")

        # consecutive same source — scale threshold with avg segs/source so long
        # versions don't get spammed for purely chronological reasons.
        from collections import Counter
        per_video = Counter(s["video"] for s in segs)
        avg_per_video = (len(segs) / max(len(per_video), 1)) if segs else 0
        eff_max = max(max_same_source, int(avg_per_video * 1.5))
        run = 1
        for i in range(1, len(segs)):
            if segs[i]["video"] == segs[i - 1]["video"]:
                run += 1
                if run > eff_max:
                    report.add(level="warn", rule="same_source_run", plan_index=i,
                               message=f"{label}[{i}]: {segs[i]['video']} {run} segs in a row "
                                       f"(threshold {eff_max} = max({max_same_source}, avg×1.5))")
            else:
                run = 1

        # adjacent similarity check
        embs_cache: dict[tuple[str, float], np.ndarray] = {}
        def emb_for(seg: dict):
            key = (seg["video"], (seg["start"] + seg["end"]) / 2)
            if key not in embs_cache:
                e = _load_emb(paths, seg["video"], key[1])
                embs_cache[key] = e
            return embs_cache[key]

        for i in range(1, len(segs)):
            a = emb_for(segs[i - 1]); b = emb_for(segs[i])
            if a is None or b is None:
                continue
            sim = cosine_sim(a, b)
            if sim > adj_sim_max:
                report.add(level="warn", rule="adjacent_similar", plan_index=i,
                           message=f"{label}[{i-1},{i}]: adjacent CLIP sim {sim:.2f} > {adj_sim_max}")
    return report


def fmt_report(report: LintReport) -> str:
    lines = []
    for f in report.findings:
        tag = {"info": "INFO ", "warn": "WARN ", "error": "ERROR"}[f.level]
        lines.append(f"{tag} [{f.rule}] {f.message}")
    return "\n".join(lines)


def run(paths: Paths, cfg: dict) -> None:
    """CLI hook: lint the existing plan.json."""
    plan_path = paths.work_dir / "plan.json"
    if not plan_path.exists():
        log("no plan.json — run `icevideo select` first", "lint")
        return
    plans = load_json(plan_path)
    report = check_plan(plans, paths, cfg)
    text = fmt_report(report)
    if text:
        print(text)
    log(f"errors={len(report.errors)}  warns={len(report.warns)}  total findings={len(report.findings)}", "lint")
