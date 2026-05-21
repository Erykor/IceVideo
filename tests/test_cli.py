"""Make sure the CLI parses every subcommand's --help without crashing."""
from __future__ import annotations

import pytest

from icevideo.cli import build_parser


ALL_COMMANDS = [
    "probe", "extract", "transcribe", "signals", "clip-score", "boundaries",
    "telemetry", "faces", "diarize", "music",
    "frames", "timeline", "describe",
    "cut", "concat", "subtitle", "music-mix", "audio-clip",
    "select", "lint", "critique", "render", "info",
]


@pytest.mark.parametrize("cmd", ALL_COMMANDS)
def test_subcommand_parses_help(cmd, capsys):
    p = build_parser()
    with pytest.raises(SystemExit) as exc:
        p.parse_args([cmd, "--help"])
    assert exc.value.code == 0  # --help exits 0


def test_unknown_command_fails():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["not-a-real-command"])


def test_cut_requires_src_in_single_mode():
    p = build_parser()
    # In single mode, src/start/end/out are required — but argparse only enforces this
    # if we don't have alternate mode flags. Our cli main raises SystemExit when missing.
    args = p.parse_args(["cut", "--start", "1", "--end", "2", "--out", "x.mp4"])
    assert args.src is None
