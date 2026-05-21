"""Run a VLM (Claude/Anthropic) over an image and print the description.

Pure helper — the AI doesn't need this if it can read the image directly. Use it for
batch scanning: describing 50 frames at once where opening each in chat is expensive.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path


DEFAULT_PROMPT = (
    "Describe what is happening in this single still frame in 1-2 sentences. "
    "Focus on subjects, action, mood, lighting. Skip generic preamble like 'this image shows'."
)


def describe(image_path: Path, *, prompt: str = DEFAULT_PROMPT, model: str | None = None) -> str:
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "`describe` needs the [critique] extra: uv sync --extra critique"
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "image/jpeg")

    content = [
        {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                     "data": base64.b64encode(image_path.read_bytes()).decode("ascii")}},
        {"type": "text", "text": prompt},
    ]
    resp = client.messages.create(
        model=model or "claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text if resp.content else ""


def cli(image_paths: list[Path], *, prompt: str | None = None, model: str | None = None) -> None:
    for img in image_paths:
        text = describe(img, prompt=prompt or DEFAULT_PROMPT, model=model)
        print(f"--- {img}")
        print(text)
        print()
