"""Step 4.7: detect faces, cluster identities, write per-second protagonist signal.

We try `insightface` first (best quality), then fall back to OpenCV's bundled Haar
cascade for *detection only* (no identity clustering, so output is just per-second
"face area ratio"). Either output is consumed by `select` as a small bonus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from icevideo.config import Paths, discover_videos
from icevideo.utils import log, save_json, video_basename


def _try_insightface(device: str = "cpu"):
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        app = FaceAnalysis(name="buffalo_sc", providers=providers)
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))
        return app
    except Exception as e:
        log(f"insightface unavailable ({e}); falling back to opencv haar", "faces")
        return None


def _opencv_haar():
    import cv2  # type: ignore
    p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not p.exists():
        return None
    return cv2.CascadeClassifier(str(p))


def _cluster_embeddings(all_embs: list[np.ndarray], threshold: float = 0.5) -> list[int]:
    """Very simple online clustering. Returns identity label per embedding."""
    if not all_embs:
        return []
    centroids: list[np.ndarray] = []
    labels: list[int] = []
    for e in all_embs:
        en = e / (np.linalg.norm(e) + 1e-9)
        if not centroids:
            centroids.append(en); labels.append(0); continue
        sims = [float(np.dot(en, c)) for c in centroids]
        idx = int(np.argmax(sims))
        if sims[idx] > 1 - threshold:
            labels.append(idx)
            # running average
            centroids[idx] = (centroids[idx] * 0.9 + en * 0.1)
            centroids[idx] /= np.linalg.norm(centroids[idx]) + 1e-9
        else:
            labels.append(len(centroids))
            centroids.append(en)
    return labels


def process_with_insightface(video: Path, app, n_sec: int) -> dict:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    samples_per_sec = 1.0  # 1 fps sampling for face detection
    sample_interval = int(fps / samples_per_sec)

    all_embs: list[np.ndarray] = []
    face_records: list[dict] = []  # one per detected face
    sec = -1
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % sample_interval == 0:
            sec += 1
            if sec >= n_sec: break
            faces = app.get(fr)
            for f in faces:
                area = float(((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) / (fr.shape[0] * fr.shape[1]))
                all_embs.append(f.normed_embedding)
                face_records.append({"sec": sec, "area_ratio": area, "emb_idx": len(all_embs) - 1})
        idx += 1
    cap.release()

    labels = _cluster_embeddings(all_embs)
    # Build per-second protagonist score: largest face's identity in that sec
    counts: dict[int, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    protagonist = max(counts, key=counts.get) if counts else None

    per_sec_face = np.zeros(n_sec, dtype=np.float32)
    per_sec_prot = np.zeros(n_sec, dtype=np.float32)
    for rec in face_records:
        s = rec["sec"]
        if 0 <= s < n_sec:
            per_sec_face[s] = max(per_sec_face[s], rec["area_ratio"])
            if labels[rec["emb_idx"]] == protagonist:
                per_sec_prot[s] = max(per_sec_prot[s], rec["area_ratio"])

    return {
        "engine": "insightface",
        "n_identities": len(set(labels)) if labels else 0,
        "protagonist_id": int(protagonist) if protagonist is not None else None,
        "per_sec_face_area": per_sec_face.tolist(),
        "per_sec_protagonist": per_sec_prot.tolist(),
    }


def process_with_haar(video: Path, cascade, n_sec: int) -> dict:
    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_interval = int(fps)

    per_sec_face = np.zeros(n_sec, dtype=np.float32)
    idx = 0
    sec = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % sample_interval == 0:
            sec += 1
            if sec >= n_sec: break
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
            best = 0.0
            for (x, y, w, h) in faces:
                best = max(best, (w * h) / (fr.shape[0] * fr.shape[1]))
            per_sec_face[sec] = best
        idx += 1
    cap.release()
    return {
        "engine": "haar",
        "n_identities": 0,
        "protagonist_id": None,
        "per_sec_face_area": per_sec_face.tolist(),
        "per_sec_protagonist": [0.0] * n_sec,  # no identity tracking
    }


def run(paths: Paths, cfg: dict) -> None:
    import torch  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = _try_insightface(device)
    cascade = None if app else _opencv_haar()
    if app is None and cascade is None:
        log("no face engine available; skipping", "faces")
        return

    out_dir = paths.subdir("faces")
    for v in discover_videos(paths):
        base = video_basename(v)
        out_json = out_dir / f"{base}.json"
        if out_json.exists():
            log(f"{base}: skip", "faces"); continue
        # need n_sec from signals
        sig_p = paths.subdir("signals") / f"{base}.json"
        from icevideo.utils import probe_duration, load_json
        if sig_p.exists():
            n_sec = load_json(sig_p)["seconds"]
        else:
            n_sec = int(probe_duration(v))
        try:
            if app:
                data = process_with_insightface(v, app, n_sec)
            else:
                data = process_with_haar(v, cascade, n_sec)
            data["video"] = base
            save_json(out_json, data)
            log(f"{base}: engine={data['engine']}  ids={data['n_identities']}  protagonist={data['protagonist_id']}",
                "faces")
        except Exception as e:
            save_json(out_json, {"video": base, "error": str(e)})
            log(f"{base}: error — {e}", "faces")
