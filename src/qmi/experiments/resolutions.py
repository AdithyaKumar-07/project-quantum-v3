"""Shared, side-effect-free CLI resolution parsing."""
from __future__ import annotations

RESOLUTION_GROUPS: dict[str, tuple[int, ...]] = {
    "16": (16,),
    "32": (32,),
    "240": (240,),
    "both": (16, 32),
    "native": (240,),
    "all": (16, 32, 240),
}


def parse_resolutions(value: str) -> tuple[int, ...]:
    """Resolve a CLI token to one or more supported experiment resolutions.

    Deliberately avoids dict.get(key, int(key)): Python evaluates the default
    argument eagerly, which caused 'all'/'both' to fail in earlier releases.
    """
    token = str(value).strip().lower()
    try:
        return RESOLUTION_GROUPS[token]
    except KeyError as exc:
        choices = ", ".join(RESOLUTION_GROUPS)
        raise ValueError(f"Unsupported resolution selector {value!r}; choose one of: {choices}") from exc
