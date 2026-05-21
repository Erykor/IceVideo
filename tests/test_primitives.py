"""Smoke tests for the primitives — probe, frames, cut, concat, audio-clip."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

from icevideo import audio_clip, concat, cut, frames, probe


def _dur(p: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(p),
    ]).strip())


def test_probe(sample_video: Path):
    info = probe.probe(sample_video)
    assert info["duration_s"] > 3.5
    assert info["video"]["width"] == 320
    assert info["audio"]["codec"] == "aac"


def test_frames_at(sample_video: Path, tmp_path: Path):
    paths = frames.extract_at(sample_video, [0.5, 2.0, 3.5], tmp_path / "frames")
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_frames_every(sample_video: Path, tmp_path: Path):
    times = frames._every(4.0, 1.0)
    assert times == [0.5, 1.5, 2.5, 3.5]


def test_frames_around(sample_video: Path, tmp_path: Path):
    times = frames._around(center=2.0, window=1.0, step=0.5, dur=4.0)
    assert times[0] == pytest.approx(1.0)
    assert times[-1] == pytest.approx(3.0)
    assert len(times) == 5


def test_cut_basic(sample_video: Path, tmp_path: Path):
    out = tmp_path / "seg.mp4"
    cut.cut(sample_video, start=0.5, end=2.5, out=out, width=320, height=240)
    assert _dur(out) == pytest.approx(2.0, abs=0.2)


def test_cut_slow_ramp(sample_video: Path, tmp_path: Path):
    out = tmp_path / "slow.mp4"
    cut.cut(sample_video, start=0.5, end=2.5, out=out, slow=0.5, width=320, height=240)
    # 2s source @ slow=0.5 -> 4s output
    assert _dur(out) == pytest.approx(4.0, abs=0.3)


def test_cut_audio_offset(sample_video: Path, tmp_path: Path):
    out = tmp_path / "joffset.mp4"
    cut.cut(sample_video, start=1.0, end=3.0, out=out,
            audio_offset=-0.3, width=320, height=240)
    assert _dur(out) == pytest.approx(2.0, abs=0.2)


def test_cut_snap_no_boundaries(sample_video: Path, tmp_path: Path):
    """snap-start/end without boundary data should silently fall back to requested."""
    out = tmp_path / "snap.mp4"
    s, e = cut.cut(sample_video, start=0.5, end=2.5, out=out,
                   snap_start=True, snap_end=True,
                   work_dir=tmp_path / "no_such_work", width=320, height=240)
    assert (s, e) == (0.5, 2.5)


def test_cut_batch_csv(sample_video: Path, sample_video2: Path, tmp_path: Path):
    csv_path = tmp_path / "batch.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src", "start", "end", "name"])
        w.writeheader()
        w.writerow({"src": str(sample_video),  "start": "0.5", "end": "2.0", "name": "a"})
        w.writerow({"src": str(sample_video2), "start": "0.0", "end": "1.5", "name": "b"})
    rows = cut._read_csv(str(csv_path))
    out_dir = tmp_path / "batch_out"
    paths = cut.cut_batch(rows, out_dir=out_dir, default_width=320, default_height=240)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert _dur(p) > 1.0


def test_concat_with_xfade(sample_video: Path, sample_video2: Path, tmp_path: Path):
    a = tmp_path / "a.mp4"; b = tmp_path / "b.mp4"
    cut.cut(sample_video, start=0.5, end=2.0, out=a, width=320, height=240)
    cut.cut(sample_video2, start=0.0, end=1.5, out=b, width=320, height=240)
    out = tmp_path / "joined.mp4"
    concat.concat([a, b], out, xfade=0.3)
    # 1.5s + 1.5s - 0.3s xfade overlap = 2.7s
    assert _dur(out) == pytest.approx(2.7, abs=0.3)


def test_concat_per_seam_audio_xfades(sample_video: Path, sample_video2: Path,
                                       sample_video3: Path, tmp_path: Path):
    clips = []
    for i, (src, s, e) in enumerate([
        (sample_video, 0.5, 2.0),
        (sample_video2, 0.0, 1.5),
        (sample_video3, 0.0, 1.5),
    ]):
        out = tmp_path / f"c{i}.mp4"
        cut.cut(src, start=s, end=e, out=out, width=320, height=240)
        clips.append(out)
    out = tmp_path / "joined_jl.mp4"
    # 2 seams, second seam has longer audio crossfade (J cut feel)
    concat.concat(clips, out, xfade=0.3, audio_xfades=[0.3, 0.8])
    assert _dur(out) > 3.0


def test_concat_parse_audio_xfades():
    assert concat._parse_audio_xfades(None, 2, 0.3) is None
    assert concat._parse_audio_xfades("0.5,0.8", 2, 0.3) == [0.5, 0.8]
    assert concat._parse_audio_xfades("0:0.7,1:0.4", 2, 0.3) == [0.7, 0.4]
    assert concat._parse_audio_xfades("1:0.5", 3, 0.3) == [0.3, 0.5, 0.3]


def test_audio_clip(sample_video: Path, tmp_path: Path):
    out = tmp_path / "a.wav"
    audio_clip.audio_clip(sample_video, start=0.5, end=2.0, out=out)
    assert out.exists()
    # use ffprobe to check duration
    d = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(out),
    ]).strip())
    assert d == pytest.approx(1.5, abs=0.2)


def test_audio_clip_spectrogram(sample_video: Path, tmp_path: Path):
    out = tmp_path / "spec.wav"
    audio_clip.audio_clip(sample_video, start=0.5, end=2.0, out=out)
    spec = tmp_path / "spec.png"
    audio_clip.spectrogram(sample_video, start=0.5, end=2.0, out=spec)
    assert spec.exists() and spec.stat().st_size > 0
