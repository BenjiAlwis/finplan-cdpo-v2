from __future__ import annotations

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
