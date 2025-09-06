from __future__ import annotations
import os
from typing import Dict, List
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from .prompts import GEN_SYSTEM, GEN_USER_TEMPLATE, AGG_SYSTEM, AGG_USER_TEMPLATE


def _chat(model_env: str, default_model: str = "gpt-4o-mini"):
    """
    Create a chat model using env vars:
      - OPENAI_API_KEY (required for LLM path)
      - OPENAI_BASE_URL (optional)
      - model_env: one of PLAN_MODEL / GEN_MODEL / AGG_MODEL
    Falls back to default_model if the env var is unset.
    """
    name = os.getenv(model_env) or default_model
    return init_chat_model(
        name=name,
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def call_generator(subquery: str, parents: Dict[str, str], snippets: List[str]) -> str:
    """
    LLM-based subquery answering.
    Inputs:
      - subquery: atomic question to answer
      - parents: answers from parent nodes (id -> answer)
      - snippets: top-k retrieved evidence strings
    If no OPENAI_API_KEY is set, returns a simple heuristic fallback
    (first line of the most relevant snippet).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return snippets[0].split("\n")[0][:300] if snippets else "N/A"

    chat = _chat("GEN_MODEL")
    parent_str = "\n".join([f"{k}: {v}" for k, v in parents.items()]) or "(none)"
    snip_str = "\n---\n".join(snippets[:6]) or "(no snippets)"

    user = HumanMessage(
        content=GEN_USER_TEMPLATE.format(
            subquery=subquery,
            parents=parent_str,
            snippets=snip_str,
        )
    )
    resp = chat.invoke([GEN_SYSTEM, user])
    return getattr(resp, "content", "").strip()


def call_aggregator(query: str, answers: Dict[str, str]) -> str:
    """
    LLM-based final synthesis of the overall answer from solved subqueries.
    If no OPENAI_API_KEY is set, returns a compact heuristic summary.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    ans_str = "\n".join([f"{k}: {v}" for k, v in sorted(answers.items())])

    if not api_key:
        return f"{query}\n\nSummary:\n{ans_str}"

    chat = _chat("AGG_MODEL", default_model=os.getenv("GEN_MODEL") or "gpt-4o-mini")
    user = HumanMessage(
        content=AGG_USER_TEMPLATE.format(
            query=query,
            answers=ans_str,
        )
    )
    resp = chat.invoke([AGG_SYSTEM, user])
    return getattr(resp, "content", "").strip()
