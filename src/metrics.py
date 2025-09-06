from __future__ import annotations
from typing import Optional

def _to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip().replace(",", "").replace("$", "")
    if not t:
        return None
    is_pct = t.endswith("%")
    if is_pct: t = t[:-1].strip()
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        return float(t)
    except Exception:
        return None

def numeric_match(pred: str, gold: str, abs_tol: float = 1e-4, rel_tol: float = 0.01) -> bool:
    p = _to_float(pred); g = _to_float(gold)
    if p is None or g is None:
        return str(pred).strip().lower() == str(gold).strip().lower()
    if abs(p - g) <= abs_tol:
        return True
    denom = max(1e-9, abs(g))
    return abs(p - g) / denom <= rel_tol
