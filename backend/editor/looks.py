"""Film looks built from formulas (FFmpeg colour filters), no LUT files needed."""
LOOKS: dict[str, str] = {
    "warm": "colorbalance=rs=0.08:gs=0.02:bs=-0.08:rm=0.05:bm=-0.05,eq=saturation=1.08",
    "cool": "colorbalance=rs=-0.06:bs=0.08:rm=-0.04:bm=0.06",
    "teal-orange": "colorbalance=rs=-0.10:bs=0.12:rh=0.10:bh=-0.10,eq=contrast=1.08:saturation=1.12",
    "vintage": "curves=preset=vintage,eq=saturation=0.85,vignette=PI/5",
    "black-white": "hue=s=0,eq=contrast=1.15",
    "game-punch": "eq=contrast=1.18:saturation=1.35:gamma=0.95,unsharp=5:5:0.6",
    "soft-film": "eq=contrast=0.94:saturation=0.9:brightness=0.02,curves=all='0/0.04 1/0.97'",
}


def filter_for(name: str | None) -> str | None:
    if not name:
        return None
    if name not in LOOKS:
        raise ValueError(f"unknown look '{name}'. Available: {sorted(LOOKS)}")
    return LOOKS[name]
