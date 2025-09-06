from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set

@dataclass
class PlanNode:
    id: str         # "i.j" tag
    text: str       # atomic subquery
    depth: int      # i
    deps: List[str] = field(default_factory=list)  # parent ids (AI.J)

@dataclass
class PlanDAG:
    nodes: Dict[str, PlanNode]

    def ready(self, answered: Set[str]) -> List[PlanNode]:
        r = []
        for n in self.nodes.values():
            if n.id in answered:
                continue
            if all(p in answered for p in n.deps):
                r.append(n)
        return sorted(r, key=lambda n: (n.depth, n.id))

    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

def heuristic_plan(question: str) -> PlanDAG:
    q = question.lower()
    nodes: Dict[str, PlanNode] = {}

    # Depth 1: grounding
    nodes["1.1"] = PlanNode(
        id="1.1",
        text="Identify the entity and fiscal period(s) referenced in the question.",
        depth=1, deps=[]
    )

    # Depth 2: metric lookup(s)
    wanted: List[str] = []
    if any(k in q for k in ["net income", "profit", "earnings"]):
        wanted.append("net income")
    if any(k in q for k in ["revenue", "sales", "top line"]):
        wanted.append("revenue")
    if any(k in q for k in ["gross margin", "operating margin", "margin"]):
        wanted.append("margin")

    if not wanted:
        wanted = ["the requested financial metric in the question"]

    for j, metric in enumerate(wanted, start=1):
        nid = f"2.{j}"
        nodes[nid] = PlanNode(
            id=nid,
            text=f"Retrieve the value for {metric} for the identified period(s).",
            depth=2, deps=["1.1"]
        )

    # Depth 3: change/delta if needed
    if any(k in q for k in ["yoy", "year over year", "change", "difference", "delta", "increase", "decrease"]):
        nodes["3.1"] = PlanNode(
            id="3.1",
            text="Compute absolute and percentage change vs prior comparable period for requested metric(s).",
            depth=3, deps=[n for n in nodes if n.startswith("2.")]
        )

    return PlanDAG(nodes=nodes)
