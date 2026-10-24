"""Run the evaluation set and write a report.

    python -m evals.run --provider ollama --model qwen2.5:7b
    python -m evals.run --provider anthropic
    python -m evals.run --provider oracle          # harness self-test, no model needed

Writes ``evals/results/<provider>-<model>.md`` (a readable report) and a JSON
file with every answer and tool call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from copilot.agent import Copilot
from copilot.providers import Provider, make_provider
from copilot.warehouse import Warehouse
from evals.grading import grade
from evals.oracle import DontKnowProvider, OracleProvider

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.yaml"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    questions = yaml.safe_load(path.read_text())
    wh = Warehouse()
    for q in questions:
        if q.get("reference_sql"):
            q["expected"] = wh._run(q["reference_sql"])["rows"][0][0]
    return questions


async def evaluate(provider: Provider, questions: list[dict]) -> dict:
    rows = []
    async with Copilot(provider) as copilot:
        for q in questions:
            try:
                res = await copilot.ask(q["question"])
                answer, trace, seconds, turns = res.answer, res.trace, res.seconds, res.turns
                error = None
            except Exception as exc:  # noqa: BLE001 - a crash counts as a failed answer
                answer, trace, seconds, turns, error = "", [], 0.0, 0, f"{type(exc).__name__}: {exc}"
            passed, reason = grade(q["kind"], answer, q.get("expected"), q.get("tolerance", 0.01))
            rows.append({
                "id": q["id"], "category": q["category"], "kind": q["kind"], "question": q["question"],
                "expected": q.get("expected"), "answer": answer, "passed": passed, "reason": reason,
                "tool_calls": len(trace), "blocked_calls": sum(1 for s in trace if s.error),
                "seconds": round(seconds, 2), "turns": turns, "error": error,
                "trace": [asdict(s) for s in trace],
            })
    return summarize(provider, rows)


def summarize(provider: Provider, rows: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["passed"])
    return {
        "provider": provider.name,
        "model": provider.model,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(rows),
        "accuracy": round(sum(r["passed"] for r in rows) / len(rows), 4),
        "by_category": {c: {"passed": sum(v), "total": len(v), "accuracy": round(sum(v) / len(v), 4)}
                        for c, v in by_cat.items()},
        "avg_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
        "median_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
        "rows": rows,
    }


def to_markdown(s: dict) -> str:
    lines = [
        f"# Eval results: {s['provider']} / {s['model']}",
        "",
        f"Run {s['run_at']} on {s['questions']} questions.",
        "",
        f"**Overall: {s['accuracy'] * 100:.1f}%** "
        f"({sum(r['passed'] for r in s['rows'])}/{s['questions']}), "
        f"{s['avg_tool_calls']} tool calls per question, median {s['median_seconds']}s per question.",
        "",
        "| Category | Passed | Accuracy |",
        "|---|---|---|",
    ]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['passed']}/{v['total']} | {v['accuracy'] * 100:.0f}% |")
    lines += ["", "## Every question", "", "| | ID | Tools | Answer | Check |", "|---|---|---|---|---|"]
    for r in s["rows"]:
        ans = re.sub(r"\s+", " ", r["answer"] or r["error"] or "")[:160].replace("|", "\\|")
        lines.append(f"| {'✅' if r['passed'] else '❌'} | {r['id']} | {r['tool_calls']} | {ans} | {r['reason']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Lakehouse Copilot.")
    p.add_argument("--provider", default="ollama", help="ollama, openai, anthropic, oracle or dont-know")
    p.add_argument("--model", default=None)
    p.add_argument("--only", default=None, help="comma-separated question ids")
    args = p.parse_args()

    questions = load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.provider == "oracle":
        provider = OracleProvider(questions)
    elif args.provider == "dont-know":
        provider = DontKnowProvider()
    else:
        provider = make_provider(args.provider, args.model)

    summary = asyncio.run(evaluate(provider, questions))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stem = re.sub(r"[^a-zA-Z0-9.-]+", "-", f"{summary['provider']}-{summary['model']}")
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / f"{stem}.md").write_text(to_markdown(summary))
    print(to_markdown(summary).split("## Every question")[0])
    print(f"Report: evals/results/{stem}.md")


if __name__ == "__main__":
    main()
