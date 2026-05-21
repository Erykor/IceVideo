"""Step 2.5: speaker diarization via pyannote.

Requires HF token. If absent or pyannote isn't installed, we skip gracefully and the
downstream `select` falls back to the voiceness signal alone.
"""
from __future__ import annotations

import os
from pathlib import Path

from icevideo.config import Paths, discover_videos
from icevideo.utils import log, save_json, video_basename


def _try_pyannote():
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        log(f"pyannote not installed ({e}); skipping diarization", "diarize")
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        log("HF_TOKEN env not set — pyannote needs auth; skipping", "diarize")
        return None
    try:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        return pipe
    except Exception as e:
        log(f"pyannote.from_pretrained failed ({e}); skipping", "diarize")
        return None


def run(paths: Paths, cfg: dict) -> None:
    pipe = _try_pyannote()
    out_dir = paths.subdir("diarize")
    if pipe is None:
        # write empty markers so downstream knows we tried
        for v in discover_videos(paths):
            base = video_basename(v)
            out_json = out_dir / f"{base}.json"
            if not out_json.exists():
                save_json(out_json, {"video": base, "skipped": True, "turns": []})
        return

    audio_dir = paths.subdir("audio")
    for v in discover_videos(paths):
        base = video_basename(v)
        out_json = out_dir / f"{base}.json"
        if out_json.exists():
            log(f"{base}: skip", "diarize"); continue
        wav = audio_dir / f"{base}.wav"
        if not wav.exists():
            continue
        try:
            diar = pipe(str(wav))
            turns = [{"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
                     for t, _, spk in diar.itertracks(yield_label=True)]
            speakers = sorted({t["speaker"] for t in turns})
            save_json(out_json, {"video": base, "speakers": speakers, "turns": turns})
            log(f"{base}: speakers={len(speakers)} turns={len(turns)}", "diarize")
        except Exception as e:
            save_json(out_json, {"video": base, "error": str(e), "turns": []})
            log(f"{base}: error — {e}", "diarize")
