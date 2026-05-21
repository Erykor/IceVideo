"""Step 2: stable-whisper transcription with word-confidence filtering."""
from __future__ import annotations

import time
from pathlib import Path

from icevideo.config import Paths, discover_videos
from icevideo.utils import log, video_basename


def run(paths: Paths, cfg: dict) -> None:
    import stable_whisper  # imported lazily so plain --help doesn't pay the import cost
    import torch

    tcfg = cfg["transcribe"]
    out_dir = paths.subdir("transcripts")
    audio_dir = paths.subdir("audio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading {tcfg['model']} on {device}", "transcribe")
    t0 = time.time()
    try:
        model = stable_whisper.load_model(tcfg["model"], device=device)
        model_name = tcfg["model"]
    except Exception as e:
        log(f"failed to load {tcfg['model']}: {e}; falling back to base", "transcribe")
        model = stable_whisper.load_model("base", device=device)
        model_name = "base"
    log(f"model loaded in {time.time()-t0:.1f}s ({model_name})", "transcribe")

    lang = None if tcfg.get("language", "auto") == "auto" else tcfg["language"]
    threshold = float(tcfg.get("word_conf_threshold", 0.4))

    for video in discover_videos(paths):
        base = video_basename(video)
        wav = audio_dir / f"{base}.wav"
        if not wav.exists():
            log(f"{base}: no audio — run `icevideo extract` first", "transcribe")
            continue
        json_out = out_dir / f"{base}.json"
        srt_out = out_dir / f"{base}.srt"
        if json_out.exists() and srt_out.exists():
            log(f"{base}: skip", "transcribe")
            continue

        t0 = time.time()
        result = model.transcribe(
            str(wav),
            language=lang,
            word_timestamps=True,
            vad=tcfg.get("vad", False),
            verbose=False,
            suppress_silence=True,
            condition_on_previous_text=False,
        )

        # Drop low-confidence words; drop segments that lose all of theirs.
        kept = dropped = 0
        new_segments = []
        for seg in result.segments:
            new_words = []
            for w in seg.words or []:
                prob = getattr(w, "probability", 1.0)
                if prob is None or prob >= threshold:
                    new_words.append(w); kept += 1
                else:
                    dropped += 1
            if new_words:
                seg.words = new_words
                seg.start = new_words[0].start
                seg.end = new_words[-1].end
                new_segments.append(seg)
        result.segments = new_segments

        result.save_as_json(str(json_out))
        try:
            result.to_srt_vtt(str(srt_out), word_level=False)
        except Exception as e:
            log(f"{base}: SRT save skipped — {e}", "transcribe")
        log(
            f"{base}: {time.time()-t0:.1f}s  lang={result.language}  "
            f"segs={len(result.segments)}  words={kept}/{kept+dropped}",
            "transcribe",
        )
