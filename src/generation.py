from __future__ import annotations
import os, json, re
from typing import Dict, List, Tuple
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from .prompts import (
    GEN_SYSTEM, GEN_USER_TEMPLATE,
    AGG_SYSTEM, AGG_USER_TEMPLATE,
    RESOLVER_SYSTEM, RESOLVER_USER_TEMPLATE,
    VALIDATOR_SYSTEM, VALIDATOR_USER_TEMPLATE
)

def _chat(model_env: str, default_model: str = "gpt-4o-mini"):
    name = os.getenv(model_env) or default_model
    return init_chat_model(
        name=name,
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

def _parse_json(text: str) -> Dict:
    # tolerant JSON extraction
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        return json.loads(fence.group(1))
    obj = re.search(r"(\{.*\})", text, flags=re.S)
    if obj:
        return json.loads(obj.group(1))
    return {}

# -------- LLM subquery generator --------
def call_generator(subquery: str, parents: Dict[str, str], snippets: List[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return snippets[0].split("\n")[0][:300] if snippets else "N/A"
    chat = _chat("GEN_MODEL")
    parent_str = "\n".join([f"{k}: {v}" for k, v in parents.items()]) or "(none)"
    snip_str = "\n---\n".join(snippets[:6]) or "(no snippets)"
    user = HumanMessage(content=GEN_USER_TEMPLATE.format(subquery=subquery, parents=parent_str, snippets=snip_str))
    resp = chat.invoke([GEN_SYSTEM, user])
    return getattr(resp, "content", "").strip()

# -------- LLM aggregator --------
def call_aggregator(query: str, answers: Dict[str, str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    ans_str = "\n".join([f"{k}: {v}" for k, v in sorted(answers.items())])
    if not api_key:
        return f"{query}\n\nSummary:\n{ans_str}"
    chat = _chat("AGG_MODEL", default_model=os.getenv("GEN_MODEL") or "gpt-4o-mini")
    user = HumanMessage(content=AGG_USER_TEMPLATE.format(query=query, answers=ans_str))
    resp = chat.invoke([AGG_SYSTEM, user])
    return getattr(resp, "content", "").strip()

# -------- NEW: LLM resolver --------
def call_resolver(query: str, memory: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # no-op fallback
        return query, {}
    chat = _chat("GEN_MODEL")  # reuse generator model
    user = HumanMessage(content=RESOLVER_USER_TEMPLATE.format(query=query, memory=json.dumps(memory, ensure_ascii=False)))
    resp = chat.invoke([RESOLVER_SYSTEM, user])
    data = _parse_json(getattr(resp, "content", ""))
    rewritten = data.get("rewritten") or query
    memu = data.get("memory_updates") or {}
    # keep only known keys
    memu = {k: v for k, v in memu.items() if k in {"entity", "period", "metric", "unit"} and isinstance(v, str) and v.strip()}
    return rewritten, memu

# -------- NEW: LLM validator --------
def call_validator(query: str, final_answer: str, answers: Dict[str, str], snippets_by_node: Dict[str, List[str]]) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    snip_text = []
    for nid, snips in sorted(snippets_by_node.items()):
        joined = " | ".join(snips[:2])
        snip_text.append(f"{nid}: {joined}")
    snips = "\n".join(snip_text) or "(no snippets)"
    ans_str = "\n".join([f"{k}: {v}" for k, v in sorted(answers.items())])

    if not api_key:
        # cheap heuristic pass
        return {"verdict": "pass", "corrected": "", "confidence": 0.6, "rationale": "No LLM; heuristic pass."}

    chat = _chat("GEN_MODEL")  # reuse generator model
    user = HumanMessage(content=VALIDATOR_USER_TEMPLATE.format(query=query, final=final_answer, answers=ans_str, snippets=snips))
    resp = chat.invoke([VALIDATOR_SYSTEM, user])
    data = _parse_json(getattr(resp, "content", ""))
    # sanitize
    verdict = str(data.get("verdict", "pass")).lower()
    if verdict not in {"pass", "fail"}:
        verdict = "pass"
    corrected = str(data.get("corrected", "") or "")
    conf = float(data.get("confidence", 0.6))
    rationale = str(data.get("rationale", ""))
    return {"verdict": verdict, "corrected": corrected, "confidence": conf, "rationale": rationale}
