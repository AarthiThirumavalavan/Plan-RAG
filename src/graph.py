from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver

from .planrag import PlanDAG, call_planner
from .generation import call_generator, call_aggregator, call_resolver, call_validator
from .numeric_tools import maybe_compute_change

class GraphState(BaseModel):
    # inputs
    query: str
    # conversation memory / slots
    memory: Dict[str, str] = Field(default_factory=dict)   # entity, period, metric, unit, etc.
    # runtime
    dag: Optional[PlanDAG] = None
    answers: Dict[str, str] = Field(default_factory=dict)
    retrieved: Dict[str, List[str]] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    validation: Dict[str, str] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)

def _merge_memory(old: Dict[str, str], updates: Dict[str, str]) -> Dict[str, str]:
    out = dict(old)
    for k, v in updates.items():
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out

# -------- Nodes --------

def resolve_node(state: GraphState) -> GraphState:
    """Resolve references/ellipsis with memory; rewrite question and update memory slots."""
    rewritten, mem_updates = call_resolver(state.query, state.memory)
    state.logs.append(f"Resolver: rewritten question: {rewritten}")
    state.query = rewritten
    state.memory = _merge_memory(state.memory, mem_updates)
    return state

def planner_node(state: GraphState) -> GraphState:
    dag = call_planner(state.query)
    state.dag = dag
    state.logs.append(f"Planner: {len(dag.nodes)} nodes; max depth={dag.max_depth()}")
    return state

def make_batch_node(retriever, k_docs: int = 6, max_workers: int = 4):
    def batch_node(state: GraphState) -> GraphState:
        assert state.dag is not None
        answered = set(state.answers.keys())
        ready = state.dag.ready(answered)
        if not ready:
            return state

        current_depth = min(n.depth for n in ready)
        batch = [n for n in ready if n.depth == current_depth]
        state.logs.append(f"Batch: depth {current_depth} with {len(batch)} node(s)")

        def _solve(node):
            parents = {pid: state.answers[pid] for pid in node.depends_on if pid in state.answers}
            hits = retriever.query(node.text, k=k_docs)
            snips = [h[0] for h in hits]
            ans = call_generator(subquery=node.text, parents=parents, snippets=snips)
            return node.id, snips, ans

        if max_workers and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(_solve, n) for n in batch]
                for fut in as_completed(futs):
                    nid, sn, an = fut.result()
                    state.retrieved[nid] = sn
                    state.answers[nid] = an
                    state.logs.append(f"Solved node {nid}")
        else:
            for n in batch:
                nid, sn, an = _solve(n)
                state.retrieved[nid] = sn
                state.answers[nid] = an
                state.logs.append(f"Solved node {nid}")

        # Opportunistic memory update from 1.1 grounding node
        if "1.1" in state.answers:
            # simple heuristics: try to capture period (YYYY) and entity (first capitalized token)
            import re
            g = state.answers["1.1"]
            year = re.search(r"(20\d{2}|19\d{2})", g or "")
            ent = re.search(r"\b([A-Z][A-Za-z0-9.&-]+(?:\s+[A-Z][A-Za-z0-9.&-]+)?)\b", g or "")
            updates = {}
            if year: updates["period"] = year.group(1)
            if ent: updates["entity"] = ent.group(1)
            if updates:
                state.memory = _merge_memory(state.memory, updates)
        return state
    return batch_node

def compute_node(state: GraphState) -> GraphState:
    """Perform numeric ops deterministically (YoY delta/%) when plan suggests it."""
    # If '3.1' exists, try to compute change from metric nodes
    computed = maybe_compute_change(state.answers)
    if computed:
        state.answers["3.1"] = computed
        state.logs.append(f"Compute: 3.1 = {computed}")
    return state

def final_node(state: GraphState) -> GraphState:
    state.final_answer = call_aggregator(state.query, state.answers) if state.answers else "No answer."
    state.logs.append("Final: synthesized.")
    return state

def validate_node(state: GraphState) -> GraphState:
    res = call_validator(state.query, state.final_answer or "", state.answers, state.retrieved)
    state.validation = {k: (str(v) if not isinstance(v, str) else v) for k, v in res.items()}
    if res.get("verdict") == "fail" and res.get("corrected"):
        state.logs.append("Validator: corrected final answer applied.")
        state.final_answer = str(res["corrected"])
    else:
        state.logs.append("Validator: pass.")
    return state

# -------- Graph builder --------

def is_complete(state: GraphState) -> bool:
    if not state.dag or not state.dag.nodes:
        return False
    return set(state.dag.nodes.keys()).issubset(state.answers.keys())

def build_graph(retriever):
    """
    START → resolve → planner → (batch loop) → compute → final → validate → END
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("resolve", resolve_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("batch", make_batch_node(retriever))
    workflow.add_node("compute", compute_node)
    workflow.add_node("final", final_node)
    workflow.add_node("validate", validate_node)

    def router_after_planner(state: GraphState) -> str:
        return "final" if is_complete(state) else "batch"

    def router_batch(state: GraphState) -> str:
        return "final" if is_complete(state) else "batch"

    workflow.add_edge(START, "resolve")
    workflow.add_edge("resolve", "planner")
    workflow.add_conditional_edges("planner", router_after_planner, {"batch": "batch", "final": "final"})
    workflow.add_conditional_edges("batch", router_batch, {"batch": "batch", "final": "final"})
    workflow.add_edge("final", "validate")   # always run validator
    workflow.add_edge("validate", END)
    # compute always before final, but only when batch completes
    workflow.add_edge("batch", "compute")
    workflow.add_edge("compute", "final")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
