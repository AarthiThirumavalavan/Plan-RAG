from __future__ import annotations
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from .retrieval import PerDocRetriever
from .planrag import PlanDAG, call_planner
from .generation import call_generator, call_aggregator

class PlanRAGRunner:
    """
    Executes a Plan*RAG DAG with LLM-based generation per node and LLM aggregation.
    Falls back to simple heuristics if no OPENAI_API_KEY is present.
    """

    def __init__(self, retriever: PerDocRetriever, max_workers: int = 4, k_docs: int = 6):
        self.retriever = retriever
        self.max_workers = max_workers
        self.k_docs = k_docs

    def run(self, question: str) -> Tuple[str, Dict[str, str], Dict[str, List[str]]]:
        """
        Orchestrates:
          1) LLM plan (PlanDAG)
          2) Depth-by-depth execution; nodes at same depth can run in parallel
          3) LLM aggregation to produce final answer

        Returns:
          (final_answer, answers_by_node, snippets_by_node)
        """
        dag: PlanDAG = call_planner(question)
        answers: Dict[str, str] = {}
        retrieved: Dict[str, List[str]] = {}

        # Keep executing until all nodes answered
        while True:
            ready = dag.ready(set(answers.keys()))
            if not ready:
                break

            # Execute one depth at a time (parallel across nodes at that depth)
            current_depth = min(n.depth for n in ready)
            batch = [n for n in ready if n.depth == current_depth]

            def _solve(node):
                # Gather parent answers for context
                parents = {pid: answers[pid] for pid in node.depends_on if pid in answers}
                # Retrieve top-k evidence snippets (restricted to this document)
                hits = self.retriever.query(node.text, k=self.k_docs)
                snips = [h[0] for h in hits]
                # LLM generation per node (or heuristic fallback inside call_generator)
                ans = call_generator(subquery=node.text, parents=parents, snippets=snips)
                return node.id, snips, ans

            if self.max_workers and self.max_workers > 1 and len(batch) > 1:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futs = [pool.submit(_solve, node) for node in batch]
                    for fut in as_completed(futs):
                        nid, sn, an = fut.result()
                        retrieved[nid] = sn
                        answers[nid] = an
            else:
                for node in batch:
                    nid, sn, an = _solve(node)
                    retrieved[nid] = sn
                    answers[nid] = an

        # Final aggregation via LLM (with internal fallback)
        final = call_aggregator(question, answers) if answers else "No answer."
        return final, answers, retrieved
