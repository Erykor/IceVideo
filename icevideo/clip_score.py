"""Step 4: CLIP semantic scores (per-prompt similarity + per-frame embeddings)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from icevideo.config import Paths
from icevideo.utils import log, save_json


def run(paths: Paths, cfg: dict) -> None:
    import torch  # lazy
    import open_clip  # lazy
    from PIL import Image  # lazy

    ccfg = cfg["clip"]
    out_dir = paths.subdir("clip_scores")
    emb_dir = paths.subdir("clip_embs")
    fr_dir = paths.subdir("clip_frames")

    positive = list(ccfg["positive_prompts"])
    negative = list(ccfg["negative_prompts"])
    prompts = positive + negative
    pos_keys = [f"pos_{i}" for i in range(len(positive))]
    neg_keys = [f"neg_{i}" for i in range(len(negative))]
    keys = pos_keys + neg_keys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading CLIP {ccfg['model']} / {ccfg['pretrained']} on {device}", "clip")
    model, _, preprocess = open_clip.create_model_and_transforms(
        ccfg["model"], pretrained=ccfg["pretrained"], device=device)
    tokenizer = open_clip.get_tokenizer(ccfg["model"])
    model.eval()
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    for vdir in sorted(fr_dir.iterdir()):
        if not vdir.is_dir():
            continue
        base = vdir.name
        out_json = out_dir / f"{base}.json"
        emb_npy = emb_dir / f"{base}.npy"
        if out_json.exists() and emb_npy.exists():
            log(f"{base}: skip", "clip")
            continue

        files = sorted(vdir.glob("*.jpg"))
        if not files:
            log(f"{base}: no frames", "clip")
            continue

        batch_size = 32
        all_sims, all_embs = [], []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
            with torch.no_grad():
                feats = model.encode_image(ims)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sims = (feats @ text_feats.T).cpu().numpy()
            all_sims.append(sims)
            all_embs.append(feats.cpu().numpy())

        sims = np.concatenate(all_sims, axis=0)
        embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        # frame i corresponds roughly to t = (i+1) / clip_fps seconds
        clip_fps = cfg["signals"]["clip_fps"]
        frame_times = [round((i + 1) / clip_fps, 2) for i in range(len(files))]

        save_json(out_json, {
            "video": base,
            "keys": keys,
            "prompts": prompts,
            "n_positive": len(positive),
            "n_negative": len(negative),
            "frame_times": frame_times,
            "sims": sims.tolist(),
        })
        np.save(emb_npy, embs)
        maxes = sims.max(axis=0)
        msg = "  ".join(f"{k}:{v:.2f}" for k, v in zip(keys, maxes))
        log(f"{base}: {msg}", "clip")
