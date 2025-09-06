from __future__ import annotations
import os
import typer
from rich import print as rprint
from rich.table import Table
from typing import Optional

from .dataset import load_records, index_by_id, get_turn_question, get_turn_gold
from .retrieval import PerDocRetriever
from .executor import PlanRAGRunner
from .metrics import numeric_match

# CLI app (script entrypoint is configured as "main = src.main:app")  :contentReference[oaicite:2]{index=2}
# Usage follows your README's pattern: uv run main chat <record_id>  :contentReference[oaicite:3]{index=3}
app = typer.Typer(help="Plan*RAG Conversational QA over ConvFinQA")

def _load(data_path: str):
    if not os.path.exists(data_path):
        rprint(f"[red]Dataset not found at {data_path}[/red]")
        raise typer.Exit(code=2)
    records = load_records(data_path)
    return index_by_id(records)

@app.command()
def chat(
    record_id: str = typer.Argument(..., help="Record ID from the dataset"),
    data: str = typer.Option("convfinqa_dataset.json", help="Path to convfinqa_dataset.json"),
    turn: Optional[int] = typer.Option(None, help="Dialogue turn index (default: last)"),
    show_snippets: bool = typer.Option(True, help="Print retrieved snippets per subquery"),
):
    """Answer one conversation turn using a Plan*RAG-style DAG executor."""
    idx = _load(data)
    if record_id not in idx:
        rprint(f"[red]Unknown record_id: {record_id}[/red]")
        raise typer.Exit(code=2)

    rec = idx[record_id]
    question = get_turn_question(rec, turn)
    if not question:
        rprint("[yellow]No question available for this record/turn.[/yellow]")
        raise typer.Exit(code=1)

    rprint(f"[bold]Question:[/bold] {question}")
    retriever = PerDocRetriever(rec)
    runner = PlanRAGRunner(retriever)

    final, answers, retrieved = runner.run(question)

    # Pretty print plan answers
    t = Table(title="Plan*RAG sub-answers")
    t.add_column("Node"); t.add_column("Answer")
    for k, v in sorted(answers.items()):
        t.add_row(k, str(v))
    rprint(t)

    if show_snippets:
        rprint("[cyan]Top snippets (by node):[/cyan]")
        for nid, snips in sorted(retrieved.items()):
            rprint(f"[bold]{nid}[/bold]")
            for s in snips[:3]:
                rprint("-", s)

    rprint(f"[bold green]Final:[/bold green] {final}")

    gold = get_turn_gold(rec, turn)
    if gold is not None:
        ok = numeric_match(final, str(gold))
        rprint(f"[bold]Gold:[/bold] {gold}    Match: {'✅' if ok else '❌'}")

@app.command()
def eval(
    data: str = typer.Option("convfinqa_dataset.json", help="Path to convfinqa_dataset.json"),
    n: int = typer.Option(50, help="Evaluate first N records × last turn"),
):
    """Tiny eval: last-turn numeric match against executed_answers (gold)."""
    idx = _load(data)
    keys = list(idx.keys())[:max(0, n)]
    total = 0
    hits = 0
    for rid in keys:
        rec = idx[rid]
        q = get_turn_question(rec, turn=None)
        gold = get_turn_gold(rec, turn=None)
        if gold is None or not q:
            continue
        retriever = PerDocRetriever(rec)
        runner = PlanRAGRunner(retriever)
        final, _ans, _ret = runner.run(q)
        total += 1
        if numeric_match(final, str(gold)):
            hits += 1
    if total == 0:
        rprint("[yellow]No evaluable examples found.[/yellow]")
        raise typer.Exit(code=1)
    rprint(f"[bold]Numeric@1[/bold]: {hits}/{total} = {hits/total:.2%}")

@app.command()
def report_template():
    """Print where to write your findings (REPORT.md)."""
    rprint("Fill in your findings in REPORT.md (template is already in the repo).")  # :contentReference[oaicite:4]{index=4}
