"""Ask the copilot a question from the command line.

    python -m copilot.ask "Which product category made the most revenue?"
    python -m copilot.ask --provider anthropic "Why are events being rejected?"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from .agent import Copilot
from .providers import make_provider


async def _main(args) -> None:
    async with Copilot(make_provider(args.provider, args.model)) as copilot:
        result = await copilot.ask(args.question)
    if args.trace:
        for i, step in enumerate(result.trace, 1):
            status = f"BLOCKED: {step.error}" if step.error else "ok"
            print(f"[{i}] {step.tool}({json.dumps(step.arguments)}) -> {status}")
        print()
    print(result.answer)
    print(f"\n({len(result.trace)} tool calls, {result.seconds:.1f}s, {args.provider})")


def main() -> None:
    p = argparse.ArgumentParser(description="Ask Lakehouse Copilot a question.")
    p.add_argument("question")
    p.add_argument("--provider", default="ollama", help="ollama (default), openai or anthropic")
    p.add_argument("--model", default=None)
    p.add_argument("--trace", action="store_true", help="show every tool call")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
