"""Read and write .srt subtitle files."""
from __future__ import annotations

import re
from dataclasses import dataclass

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def fmt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_time(text: str) -> float:
    m = _TIME.search(text)
    if not m:
        raise ValueError(f"bad time: {text}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def parse(content: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip().lstrip("\ufeff")):
        lines = [l for l in block.splitlines() if l.strip()]
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->")
        cues.append(Cue(parse_time(a), parse_time(b), "\n".join(lines[idx + 1:])))
    return cues


def dump(cues: list[Cue]) -> str:
    parts = [f"{i}\n{fmt_time(c.start)} --> {fmt_time(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)]
    return "\n".join(parts)
