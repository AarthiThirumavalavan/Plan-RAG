from __future__ import annotations
from typing import Dict, List, Tuple
import re
from .retrieval import PerDocRetriever
from .planrag import PlanDAG, heuristic_plan, PlanNode

def _first_number(text: str) -> str | None:
    # Pull a canonical-looking number (handles commas, %, parentheses)
    t = text.replace(",", "")
    # capture neg in parentheses
    if t.strip().startswith("(") and t.strip().endswith(")"):
        t = "-" + t.strip()[1:-1]
    m = re.search(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?%?", t, flags=re.I)
    return m.group(0) if m else None

def _compute_change(curr: str, prev: str) -> str:
    try:
        def to_float(s: str) -> float:
            s = s.strip().replace(",", "").replace("$", "")
            pct = s.endswith("%")
            if pct: s = s[:-1]
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]
            return float(s)
        c = to_float(curr); p = to_float(prev)
        delta = c - p
        rel = (delta / (p if abs(p) > 1e-9 else 1.0)) * 100.0
        return f"{delta:.4g} ({rel:.2f}%)"
    except Exception:
        return f"change_from_parents(curr={curr}, prev={prev})"

class PlanRAGRunner:
    """Simple Plan*RAG executor (LLM-free heuristic for demo)."""
    def __init__(self, retriever: PerDocRetriever):
        self.retriever = retriever

    def run(self, question: str) -> Tuple[str, Dict[str, str], Dict[str, List[str]]]:
        plan: PlanDAG = heuristic_plan(question)
        answers: Dict[str, str] = {}
        retrieved: Dict[str, List[str]] = {}

        # Iterate depth-by-depth: process all ready nodes each round
        while True:
            ready = plan.ready(set(answers.keys()))
            if not ready:
                break

            # execute one depth "in parallel" (here sequentially but grouped)
            curr_depth = min(n.depth for n in ready)
            batch = [n for n in ready if n.depth == curr_depth]

            for node in batch:
                snips = [t for (t, _s) in self.retriever.query(node.text, k=6)]
                retrieved[node.id] = snips

                # naive "generator"
                if node.depth == 1:
                    # Grounding: echo entity/period if any tokens look like years/quarters
                    years = re.findall(r"(20\d{2}|19\d{2})", " ".join(snips))
                    period = years[0] if years else "requested period"
                    answers[node.id] = f"Grounded on {period}"
                    continue

                if node.depth == 2:
                    # Extract first numeric from top snippet
                    val = None
                    for s in snips:
                        val = _first_number(s)
                        if val:
                            break
                    answers[node.id] = val or "N/A"
                    continue

                if node.depth >= 3:
                    # If we have at least two depth-2 answers, compute change
                    d2 = [answers[k] for k in sorted(answers) if k.startswith("2.") and answers[k] not in (None, "N/A")]
                    if len(d2) >= 2:
                        answers[node.id] = _compute_change(d2[0], d2[1])
                    else:
                        answers[node.id] = "insufficient_data_for_change"

        # Final aggregation: for demo, prefer latest depth answer, else join
        final = None
        if "3.1" in answers:
            final = answers["3.1"]
        elif answers:
            # last inserted key
            final = list(answers.values())[-1]
        else:
            final = "No answer."

        return final, answers, retrieved
