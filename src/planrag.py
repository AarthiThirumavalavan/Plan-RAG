from __future__ import annotations
import os, json, re
from typing import Dict, List, Set, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from .prompts import PLANNER_SYSTEM, PLANNER_USER_TEMPLATE

class PlanNode(BaseModel):
    """One DAG node: atomic subquery."""
    id: str                 # "i.j"
    text: str               # subquery text
    depth: int              # i (1-based)
    depends_on: List[str] = Field(default_factory=list)  # parent ids

class PlanDAG(BaseModel):
    """Lightweight DAG container + helpers."""
    nodes: Dict[str, PlanNode] = Field(default_factory=dict)

    def ready(self, answered: Set[str]) -> List[PlanNode]:
        """Nodes whose parents are all answered (and not answered themselves)."""
        r = []
        for n in self.nodes.values():
            if n.id in answered:
                continue
            if all(p in answered for p in n.depends_on):
                r.append(n)
        return sorted(r, key=lambda n: (n.depth, n.id))

    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

# ---------- LLM planner ----------

def _planner_chat():
    name = os.getenv("PLAN_MODEL") or "gpt-4o-mini"
    return init_chat_model(
        name=name,
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

def _parse_plan_json(text: str) -> Dict:
    """Be resilient to code fences, extra text, etc."""
    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract ```json ... ``` block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        return json.loads(fence.group(1))
    # Try to extract the first {...} blob
    curly = re.search(r"(\{.*\})", text, flags=re.S)
    if curly:
        return json.loads(curly.group(1))
    raise ValueError("Could not parse planner JSON.")

def call_planner(question: str) -> PlanDAG:
    """
    LLM-based planner: returns a PlanDAG with nodes:
      nodes=[{id, text, depth, depends_on: []}]
    Falls back to heuristic_plan on failure or missing key.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return heuristic_plan(question)

    try:
        chat = _planner_chat()
        user = HumanMessage(content=PLANNER_USER_TEMPLATE.format(query=question))
        resp = chat.invoke([PLANNER_SYSTEM, user])
        raw = getattr(resp, "content", "")
        data = _parse_plan_json(raw)

        dag = PlanDAG(nodes={})
        for node in data.get("nodes", []):
            dag.nodes[node["id"]] = PlanNode(
                id=str(node["id"]),
                text=str(node["text"]),
                depth=int(node["depth"]),
                depends_on=[str(x) for x in node.get("depends_on", [])],
            )
        if not dag.nodes:
            raise ValueError("Planner produced no nodes.")
        return dag
    except Exception:
        # Safe fallback keeps you unblocked.
        return heuristic_plan(question)

# ---------- Heuristic fallback (kept small & simple) ----------

def heuristic_plan(question: str) -> PlanDAG:
    """
    Small non-LLM plan as a backup:
      1.1: Ground entity/period
      2.*: Fetch metric(s)
      3.1: Compute change if the query hints at YoY/delta
    """
    q = question.lower()
    nodes: Dict[str, PlanNode] = {}

    nodes["1.1"] = PlanNode(
        id="1.1",
        text="Identify the entity and fiscal period(s) referenced in the question.",
        depth=1,
        depends_on=[],
    )

    metrics: List[str] = []
    if any(k in q for k in ["net income", "profit", "earnings"]):
        metrics.append("net income")
    if any(k in q for k in ["revenue", "sales", "top line"]):
        metrics.append("revenue")
    if any(k in q for k in ["gross margin", "operating margin", "margin"]):
        metrics.append("margin")
    if not metrics:
        metrics = ["the requested financial metric in the question"]

    for j, m in enumerate(metrics, start=1):
        nid = f"2.{j}"
        nodes[nid] = PlanNode(
            id=nid,
            text=f"Retrieve the value for {m} for the identified period(s).",
            depth=2,
            depends_on=["1.1"],
        )

    if any(k in q for k in ["yoy", "year over year", "change", "difference", "delta", "increase", "decrease"]):
        nodes["3.1"] = PlanNode(
            id="3.1",
            text="Compute absolute and percentage change vs prior comparable period for the requested metric(s).",
            depth=3,
            depends_on=[nid for nid in nodes if nid.startswith("2.")],
        )

    return PlanDAG(nodes=nodes)
