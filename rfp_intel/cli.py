"""Command line interface.

    python main.py --bid ./data/Bid1                 # extraction mode (one command)
    python main.py index ./data/Bid1 ./data/Bid2     # (incremental) indexing
    python main.py search "bid bond" --bid-id Bid1 --doc-type rfp
    python main.py ask "What is the submission deadline for Bid1 after all addendums?"
    python main.py qa-log eval/qa_questions.txt      # batch Q&A -> outputs/qa_log.md
    python main.py eval                              # retrieval evaluation table
    python main.py serve                             # REST API on :8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

from .config import get_settings

app = typer.Typer(add_completion=False, help="RFP Intelligence Platform")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context,
         bid: Optional[List[Path]] = typer.Option(None, "--bid", help="bid folder(s) to extract (shortcut for `extract`)"),
         verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)
    if ctx.invoked_subcommand is None:
        if bid:
            extract(bid, force=False)
        else:
            typer.echo(ctx.get_help())


@app.command()
def index(folders: List[Path], force: bool = typer.Option(False, help="re-index even if unchanged")) -> None:
    """Parse, chunk, embed and index bid folders (incremental per bid)."""
    from .search.index import SearchIndex

    idx = SearchIndex()
    for f in folders:
        rep = idx.index_folder(f, force=force)
        typer.echo(f"{rep.bid_id}: {rep.status} ({rep.documents} docs, {rep.chunks} chunks)")
        for w in rep.warnings:
            typer.echo(f"   warning: {w}")


@app.command()
def extract(folders: List[Path], force: bool = typer.Option(False, help="force re-index")) -> None:
    """Extraction mode: produce outputs/<bid>.json (+ _report.md) for each bid folder."""
    from .agents.orchestrator import Orchestrator

    orch = Orchestrator()
    if orch.llm is None:
        typer.secho("No LLM configured (set LLM_PROVIDER and its API key); cannot extract.", fg="red")
        raise typer.Exit(2)
    for f in folders:
        rec = asyncio.run(orch.extract_bid(folder=f, force_reindex=force))
        out = Path(get_settings().output_dir) / f"{rec['bid_id']}.json"
        typer.echo(f"{rec['bid_id']}: {rec['validation']} -> {out}  (trace: {rec.get('trace_file')})")
        r = rec["run"]
        typer.echo(f"   llm calls {r['llm_calls']} (cache hits {r['llm_cache_hits']}), tokens in/out "
                   f"{r['tokens_in']}/{r['tokens_out']}, est. cost ${r['est_cost_usd']}")
        for e in r["errors"]:
            typer.secho(f"   error: {e}", fg="yellow")


@app.command()
def search(query: str, bid_id: Optional[str] = None, doc_type: Optional[List[str]] = typer.Option(None),
           addendum: Optional[int] = None, top_k: int = 5, mode: str = "hybrid",
           no_expand: bool = False, no_rerank: bool = False, as_json: bool = typer.Option(False, "--json")) -> None:
    """Search the index (hybrid BM25 + vector, RRF, rerank) and print cited passages."""
    from .search.engine import SearchEngine

    filters = {k: v for k, v in {"bid_id": bid_id, "doc_type": doc_type or None, "addendum_number": addendum}.items() if v}
    hits = SearchEngine().search(query, top_k=top_k, filters=filters or None, mode=mode,  # type: ignore[arg-type]
                                 expand=not no_expand, rerank=not no_rerank)
    if as_json:
        typer.echo(json.dumps([h.to_dict() for h in hits], indent=2))
        return
    for h in hits:
        c = h.citation
        typer.secho(f"\n#{h.rank} [{h.score:.3f}] {c['file']} p.{c['page']} ({c['doc_type']}) {h.chunk.section or ''}", bold=True)
        typer.echo(h.chunk.text[:600])


@app.command()
def ask(question: str, bid_id: Optional[List[str]] = typer.Option(None)) -> None:
    """Question-answering mode: cited answer from the multi-agent system."""
    from .agents.orchestrator import Orchestrator

    res = asyncio.run(Orchestrator().answer(question, bid_id or None))
    typer.echo(res["answer"])
    for c in res.get("citations", []):
        typer.echo(f"  [{c['id']}] {c['bid_id']} / {c['file']} p.{c['page']}")


@app.command("qa-log")
def qa_log(questions_file: Path, out: Path = Path("outputs/qa_log.md")) -> None:
    """Answer every question in a file (one per line) and write a Markdown Q&A log."""
    from .agents.orchestrator import Orchestrator

    orch = Orchestrator()
    qs = [q.strip() for q in questions_file.read_text().splitlines() if q.strip() and not q.startswith("#")]
    lines = ["# Sample Q&A log", ""]
    records = []
    for i, q in enumerate(qs, 1):
        res = asyncio.run(orch.answer(q))
        records.append(res)
        lines += [f"## {i}. {q}", "", res["answer"], ""]
        for c in res.get("citations", []):
            lines.append(f"- [{c['id']}] {c['bid_id']} / {c['file']}, page {c['page']}")
        lines += ["", f"_confidence {res.get('confidence', 0):.2f}_", ""]
        typer.echo(f"{i}. {q}\n   -> {res['answer'][:200]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    out.with_suffix(".json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    typer.echo(f"wrote {out}")


@app.command("eval")
def eval_cmd(questions: Path = Path("eval/retrieval_questions.jsonl"),
             out: Path = Path("eval/retrieval_results.md")) -> None:
    """Retrieval evaluation (Recall@k, MRR, nDCG) across search configurations."""
    from .evaluation import evaluate, load_questions, to_markdown
    from .search.engine import SearchEngine

    res = evaluate(SearchEngine(), load_questions(questions))
    md = to_markdown(res)
    typer.echo(md)
    out.write_text(md + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))


@app.command()
def bids() -> None:
    """List indexed bids and their documents."""
    from .search.index import SearchIndex

    typer.echo(json.dumps(SearchIndex().catalog(), indent=2))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the REST API (FastAPI + uvicorn)."""
    import uvicorn

    uvicorn.run("rfp_intel.api.app:app", host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
