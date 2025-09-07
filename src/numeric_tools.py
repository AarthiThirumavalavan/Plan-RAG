from __future__ import annotations
import re
from typing import Optional, Tuple, Dict

_NUM = re.compile(r"[-+]?\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?%?|\d+(?:\.\d+)?%?")

def parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    t = str(s)
    m = _NUM.search(t)
    if not m:
        return None
    tok = m.group(0)
    neg = False
    if tok.startswith("(") and tok.endswith(")"):
        neg = True
        tok = tok[1:-1]
    tok = tok.replace(",", "").replace("$", "").strip()
    is_pct = tok.endswith("%")
    if is_pct:
        tok = tok[:-1].strip()
    try:
        val = float(tok)
        if is_pct:
            val = val / 100.0
        if neg:
            val = -val
        return val
    except Exception:
        return None

def compute_change(curr: float, prev: float) -> Tuple[float, float]:
    delta = curr - prev
    pct = (delta / prev) if abs(prev) > 1e-12 else float("inf")
    return delta, pct

def format_change(delta: float, pct: float) -> str:
    return f"{delta:.4g} ({pct*100:.2f}%)"

def maybe_compute_change(answers: Dict[str, str]) -> Optional[str]:
    """
    If there is a compute node '3.1' and at least two 2.* metric nodes,
    compute delta and % from the first two available metrics.
    """
    if "3.1" not in answers:
        return None
    # gather metric nodes in order
    keys = sorted([k for k in answers.keys() if k.startswith("2.")], key=lambda x: tuple(int(p) for p in x.split(".")))
    if len(keys) < 2:
        return None
    a = parse_number(answers[keys[0]])
    b = parse_number(answers[keys[1]])
    if a is None or b is None:
        return None
    delta, pct = compute_change(a, b)
    return format_change(delta, pct)
