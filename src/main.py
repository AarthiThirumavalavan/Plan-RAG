#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from typing import Optional, Dict

import typer
from rich import print as rprint
from rich.table import Table

from .dataset import load_records, index_by_id, get_turn_question, get_turn_gold
from .retrieval import PerDocHybridRetriever
from .graph import build_graph, GraphState
from .metrics import numeric_match

app = typer.Typer(help="Plan*RAG over ConvFinQA with LangGraph + Hybrid Retrieval + Memory + Validator")

def _load(data_path: str):
    if not os.path.exists(data_path):
        rprint(f"[red]Dataset not found at {data_path}[/red]")
        raise typer.Exit(code=2)
    records = load_records(data_path)
    return index_by_id(records)

def _parse_json_dict(s: Optional[str]) -> Dict[str, str]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    rprint("[yellow]Warning: --memory was not valid JSON; ignoring.[/yellow]")
    return {}

@app.command()
def chat(
    record_id: str = typer.Argument(..., help="Record ID from the dataset"),
    data: str = typer.Option("convfinqa_dataset.json", help="Path to convfinqa_dataset.json"),
    turn: Optional[int] = typer.Option(None, help="Dialogue turn index (default: last)"),
    show_snippets: bool = typer.Option(True, help="Print retrieved snippets per subquery"),
    show_logs: bool = typer.Option(False, help="Print LangGraph execution logs"),
    show_validation: bool = typer.Option(True, help="Show validator verdict/correction"),
    memory: Optional[str] = typer.Option(None, help='Initial memory JSON, e.g. \'{"entity":"Apple","period":"2020"}\''),
):
    """
    Answer one conversation turn using LangGraph Plan*RAG with:
    - conversation memory + resolver
    - hybrid retrieval (cosine dense via vector DB + BM25)
    - calculator node for numeric ops
    - answer validator/self-consistency
    """
    idx = _load(data)
    if record_id not in idx:
        rprint(f"[red]Unknown record_id: {record_id}[/red]")
        raise typer.Exit(code=2)

    rec = idx[record_id]
    question = get_turn_question(rec, turn)
    if not question:
        rprint("[yellow]No question available for this record/turn.[/yellow]")
        raise typer.Exit(code=1)

    init_memory = _parse_json_dict(memory)

    rprint(f"[bold]Question:[/bold] {question}")
    if init_memory:
        rprint(f"[dim]Initial memory:[/dim] {init_memory}")

    retriever = PerDocHybridRetriever(rec) # chunks
    app_graph = build_graph(retriever)

    result = app_graph.invoke(GraphState(query=question, memory=init_memory))

    # Sub-answers
    t = Table(title="Plan*RAG sub-answers")
    t.add_column("Node"); t.add_column("Answer")
    for k, v in sorted(result.answers.items()):
        t.add_row(k, str(v))
    rprint(t)

    # Evidence
    if show_snippets:
        rprint("[cyan]Top snippets (by node):[/cyan]")
        for nid, snips in sorted(result.retrieved.items()):
            rprint(f"[bold]{nid}[/bold]")
            for s in snips[:3]:
                rprint("-", s)

    # Final
    rprint(f"[bold green]Final:[/bold green] {result.final_answer}")

    # Validator
    if show_validation and result.validation:
        rprint(f"[bold magenta]Validator:[/bold magenta] {result.validation.get('verdict','')}"
               f" | conf={result.validation.get('confidence','')}"
               f" | reason={result.validation.get('rationale','')}")
        if result.validation.get("verdict") == "fail" and result.validation.get("corrected"):
            rprint(f"[bold]Corrected Final:[/bold] {result.validation['corrected']}")

    # Memory snapshot
    if result.memory:
        rprint(f"[dim]Memory snapshot:[/dim] {result.memory}")

    # Gold check (optional metric view)
    gold = get_turn_gold(rec, turn)
    if gold is not None:
        ok = numeric_match(result.final_answer, str(gold))
        rprint(f"[bold]Gold:[/bold] {gold}    Match: {'✅' if ok else '❌'}")

    if show_logs and result.logs:
        rprint("[dim]Logs:[/dim]")
        for line in result.logs:
            rprint("•", line)

@app.command()
def eval(
    data: str = typer.Option("convfinqa_dataset.json", help="Path to convfinqa_dataset.json"),
    n: int = typer.Option(50, help="Evaluate first N records × last turn"),
    use_memory: bool = typer.Option(True, help="Carry memory across examples (per record only)"),
):
    """
    Tiny eval: last-turn numeric match against gold.
    Optionally carries memory within a record (not across records).
    """
    idx = _load(data)
    keys = list(idx.keys())[:max(0, n)]
    total = hits = 0

    for rid in keys:
        rec = idx[rid]
        q = get_turn_question(rec, None)
        gold = get_turn_gold(rec, None)
        if gold is None or not q:
            continue

        retriever = PerDocHybridRetriever(rec)
        app_graph = build_graph(retriever)

        # per-record memory (optional)
        mem: Dict[str, str] = {} if use_memory else {}

        result = app_graph.invoke(GraphState(query=q, memory=mem))
        total += 1
        if numeric_match(result.final_answer, str(gold)):
            hits += 1

    if total == 0:
        rprint("[yellow]No evaluable examples found.[/yellow]")
        raise typer.Exit(code=1)
    rprint(f"[bold]Numeric@1[/bold]: {hits}/{total} = {hits/total:.2%}")

@app.command()
def repl(
    record_id: str = typer.Argument(..., help="Record ID from the dataset"),
    data: str = typer.Option("convfinqa_dataset.json", help="Path to convfinqa_dataset.json"),
    show_validation: bool = typer.Option(True, help="Show validator verdict each turn"),
):
    """
    Interactive loop to ask multiple questions against the same record.
    Conversation memory is preserved across turns and used by the resolver.
    """
    idx = _load(data)
    if record_id not in idx:
        rprint(f"[red]Unknown record_id: {record_id}[/red]")
        raise typer.Exit(code=2)

    rec = idx[record_id]
    rprint(f"[bold]Loaded record:[/bold] {record_id}")
    retriever = PerDocHybridRetriever(rec)
    app_graph = build_graph(retriever)

    memory: Dict[str, str] = {}
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        result = app_graph.invoke(GraphState(query=q, memory=memory))
        memory = dict(result.memory or {})  # carry forward
        print("bot>", result.final_answer)
        if show_validation and result.validation:
            print("   validator:", result.validation.get("verdict"),
                  "| conf=", result.validation.get("confidence"),
                  "| reason=", result.validation.get("rationale"))

@app.command()
def report_template():
    """Print where to write your findings (REPORT.md)."""
    rprint("Fill in your findings in REPORT.md")

if __name__ == "__main__":
    app()
