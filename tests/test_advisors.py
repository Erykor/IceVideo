"""Smoke tests for advisors — lint, critique-heuristic — with synthetic plans."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from icevideo import critique, lint as lint_mod


def _minimal_paths(tmp_path: Path):
    from icevideo.config import Paths
    work = tmp_path / "work"
    out = tmp_path / "output"
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return Paths(input_dir=tmp_path, work_dir=work, output_dir=out, input_glob="*.MP4",
                 input_dirs=[tmp_path])


def _minimal_cfg() -> dict:
    return {
        "durations": {"versions": [{"label": "1min", "seconds": 60}]},
        "scoring": {"motion": 1.0, "flow": 0.5, "audio": 0.6, "dialogue": 0.6,
                    "laughter": 1.2, "keyword": 1.0, "clip": 1.5},
        "selection": {"per_video_max": 8},
        "lint": {"min_seg": 2.0, "max_seg": 15.0,
                  "max_consecutive_same_source": 3,
                  "adjacent_sim_max": 0.93,
                  "total_duration_tolerance": 0.05},
        "critique": {"iterate_threshold": 7.0},
    }


def test_lint_flags_short_segment(tmp_path: Path):
    plans = {"1min": [
        {"video": "A", "start": 0, "end": 5, "score": 10},
        {"video": "B", "start": 0, "end": 1, "score": 5},   # 1s = too short
        {"video": "C", "start": 0, "end": 10, "score": 8},
    ]}
    paths = _minimal_paths(tmp_path); cfg = _minimal_cfg()
    rep = lint_mod.check_plan(plans, paths, cfg)
    assert any(f.rule == "short_clip" for f in rep.findings)


def test_lint_flags_long_segment(tmp_path: Path):
    plans = {"1min": [
        {"video": "A", "start": 0, "end": 25, "score": 10},
    ]}
    paths = _minimal_paths(tmp_path); cfg = _minimal_cfg()
    rep = lint_mod.check_plan(plans, paths, cfg)
    assert any(f.rule == "long_clip" for f in rep.findings)


def test_lint_flags_same_source_run(tmp_path: Path):
    # Mixed sources, but one dominates with a long consecutive run.
    # 10 segs total across 3 sources: A,A,A,A,A,A,A,A,B,C → avg=3.3, threshold=max(3,5)=5
    # A appears 8 times in a row, exceeding 5 → warn.
    plans = {"1min": [
        {"video": "A", "start": i * 5, "end": (i + 1) * 5, "score": 10} for i in range(8)
    ] + [
        {"video": "B", "start": 40, "end": 45, "score": 9},
        {"video": "C", "start": 45, "end": 50, "score": 9},
    ]}
    paths = _minimal_paths(tmp_path); cfg = _minimal_cfg()
    rep = lint_mod.check_plan(plans, paths, cfg)
    assert any(f.rule == "same_source_run" for f in rep.findings)


def test_lint_no_false_positive_single_source(tmp_path: Path):
    """When only one source exists, runs are unavoidable — should NOT warn."""
    plans = {"1min": [
        {"video": "A", "start": i * 5, "end": (i + 1) * 5, "score": 10} for i in range(8)
    ]}
    paths = _minimal_paths(tmp_path); cfg = _minimal_cfg()
    rep = lint_mod.check_plan(plans, paths, cfg)
    assert not any(f.rule == "same_source_run" for f in rep.findings)


def test_critique_heuristic_bounded(tmp_path: Path):
    plan = [
        {"video": "A", "start": 0, "end": 10, "score": 9},
        {"video": "B", "start": 0, "end": 10, "score": 7},
        {"video": "C", "start": 0, "end": 10, "score": 8},
        {"video": "D", "start": 0, "end": 10, "score": 6},
        {"video": "E", "start": 0, "end": 10, "score": 9},
        {"video": "F", "start": 0, "end": 10, "score": 5},
    ]
    paths = _minimal_paths(tmp_path); cfg = _minimal_cfg()
    r = critique.heuristic_critique("1min", plan, paths, cfg)
    s = r["scores"]
    for k in ("pacing", "hook", "narrative", "variety", "ending", "duration", "overall"):
        assert 0.0 <= s[k] <= 10.0, f"score {k}={s[k]} out of bounds"


def test_critique_iterate_flag_low_score(tmp_path: Path):
    plan = [
        {"video": "A", "start": 0, "end": 1.5, "score": 3},  # too short → pacing penalty
        {"video": "A", "start": 0, "end": 1.5, "score": 2},  # same source dominance
        {"video": "A", "start": 0, "end": 1.5, "score": 1},
    ]
    paths = _minimal_paths(tmp_path); cfg = _minimal_cfg()
    r = critique.heuristic_critique("1min", plan, paths, cfg)
    assert r["iterate"] is True
