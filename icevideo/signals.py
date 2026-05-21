"""Step 3: per-second motion (scene + optical flow) + audio (RMS, voiceness, laughter)."""
from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np

from icevideo.config import Paths, discover_videos
from icevideo.utils import (
    log, probe_duration, run_ffmpeg_with_stderr, save_json, video_basename,
)


def ffmpeg_scene_scores(video: Path) -> tuple[list[float], list[float]]:
    """Return parallel (time, score) lists from ffmpeg scene detection."""
    stderr = run_ffmpeg_with_stderr([
        "-i", str(video),
        "-vf", "select='gt(scene,0)',scale=320:180,showinfo,metadata=mode=print",
        "-an", "-f", "null", "-",
    ])
    times: list[float] = []; scores: list[float] = []
    cur_t: float | None = None
    for line in stderr.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cur_t = float(m.group(1)); continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and cur_t is not None:
            times.append(cur_t); scores.append(float(m.group(1)))
            cur_t = None
    return times, scores


def per_sec_scene(times: list[float], scores: list[float], n_sec: int) -> np.ndarray:
    arr = np.zeros(n_sec, dtype=np.float32)
    for t, s in zip(times, scores):
        i = int(t)
        if 0 <= i < n_sec:
            arr[i] = max(arr[i], s)
    return arr


def optical_flow_per_sec(video: Path, n_sec: int) -> np.ndarray:
    """Sample 1 frame/sec at 160x90 grayscale, compute Farneback flow magnitude."""
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return np.zeros(n_sec, dtype=np.float32)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    targets = {int(round(fps * s)) for s in range(n_sec)}
    flow_mag = np.zeros(n_sec, dtype=np.float32)
    prev_gray = None
    out_i = 0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in targets:
            small = cv2.resize(fr, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                if out_i < n_sec:
                    flow_mag[out_i] = float(np.percentile(mag, 90))
            out_i += 1
            prev_gray = gray
        i += 1
        if out_i >= n_sec:
            break
    cap.release()
    return flow_mag


def audio_per_sec(wav: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rms_dbfs, voiceness, laughter) per second from a 16 kHz mono WAV."""
    from scipy import stats  # type: ignore

    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n_sec = int(np.ceil(len(samples) / sr))
    rms_db = np.full(n_sec, -80.0, dtype=np.float32)
    voiceness = np.zeros(n_sec, dtype=np.float32)
    laughter = np.zeros(n_sec, dtype=np.float32)
    for i in range(n_sec):
        a = samples[i * sr:(i + 1) * sr]
        if len(a) < 256:
            continue
        r = np.sqrt(np.mean(a * a) + 1e-12)
        rms_db[i] = 20 * np.log10(r + 1e-9)
        w_len = 1024
        flats, zcrs, envs = [], [], []
        for s_i in range(0, len(a) - w_len, w_len // 2):
            win = a[s_i:s_i + w_len] * np.hanning(w_len)
            spec = np.abs(np.fft.rfft(win)) ** 2 + 1e-12
            flats.append(stats.gmean(spec) / spec.mean())
            zcrs.append(np.mean(np.abs(np.diff(np.sign(win)))) * 0.5)
            envs.append(np.sqrt(np.mean(win * win)))
        if not flats:
            continue
        mean_sfm = float(np.mean(flats))
        voiceness[i] = float(np.clip(1.0 - mean_sfm * 2.5, 0.0, 1.0))
        env_var = float(np.std(envs) / (np.mean(envs) + 1e-9))
        zcr_mean = float(np.mean(zcrs))
        if voiceness[i] > 0.4 and env_var > 0.5 and zcr_mean > 0.05:
            laughter[i] = min(1.0, env_var * 0.5 + zcr_mean * 4)
    return rms_db, voiceness, laughter


def run(paths: Paths, cfg: dict) -> None:
    out_dir = paths.subdir("signals")
    audio_dir = paths.subdir("audio")

    for video in discover_videos(paths):
        base = video_basename(video)
        out_json = out_dir / f"{base}.json"
        if out_json.exists():
            log(f"{base}: skip", "signals")
            continue

        n_sec = int(np.ceil(probe_duration(video)))
        log(f"{base}: {n_sec}s", "signals")

        times, scores = ffmpeg_scene_scores(video)
        scene = per_sec_scene(times, scores, n_sec)
        flow = optical_flow_per_sec(video, n_sec)
        wav = audio_dir / f"{base}.wav"
        if not wav.exists():
            log(f"{base}: no audio — run extract first", "signals")
            continue
        rms, voice, laugh = audio_per_sec(wav)

        n = max(n_sec, len(rms), len(voice), len(laugh), len(scene), len(flow))
        def pad(x, fill=0.0):
            x = np.asarray(x, dtype=np.float32)
            return np.concatenate([x, np.full(n - len(x), fill, dtype=np.float32)]) if len(x) < n else x[:n]

        save_json(out_json, {
            "video": base, "seconds": n,
            "scene": pad(scene).tolist(),
            "flow": pad(flow).tolist(),
            "audio_rms_db": pad(rms, -80).tolist(),
            "voiceness": pad(voice).tolist(),
            "laughter": pad(laugh).tolist(),
        })
        log(
            f"  scene[{scene.max():.2f}]  flow[{flow.max():.2f}]  "
            f"voice[{voice.max():.2f}]  laugh[{laugh.max():.2f}]",
            "signals",
        )
