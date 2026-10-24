"""The copilot agent: an LLM in a tool-calling loop, connected to the lakehouse over MCP.

    async with Copilot(provider) as copilot:
        result = await copilot.ask("Which category made the most revenue?")
        print(result.answer)

The agent starts the MCP server as a subprocess and talks to it over stdio,
exactly as Claude Desktop or Cursor would. Every tool call is recorded in
``result.trace`` so answers can be audited and evaluated.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .providers import Provider, Turn

SYSTEM_PROMPT = """You are Lakehouse Copilot, a careful data analyst for an e-commerce lakehouse.
You answer questions only with numbers you got from the tools in this conversation. Never guess.

How to work:
1. For business numbers (revenue, orders, conversion, customers, rejected events) call list_metrics,
   then get_metric. Metrics have agreed definitions, so prefer them over your own SQL.
2. Otherwise call list_tables and describe_table, then run_sql with one DuckDB SELECT that uses
   schema-qualified tables (for example gold.fct_orders).
3. For data quality or pipeline questions use pipeline_health and contract_violations.
4. If a tool returns an error, read it, fix your call and try again.

How to answer:
- Be brief. Lead with the answer, then say which metric or table it came from.
- Write rates as percentages (0.1873 is 18.73%). Round money to cents.
- The lakehouse is read-only. If you are asked to change, delete, insert or create data, or to read
  files or settings, do not attempt it: say you only have read-only access.
- If the data cannot answer the question (the field or time period does not exist), say that
  plainly instead of estimating. Check which dates exist before saying a period has no data.
"""

MAX_RESULT_CHARS = 6000


@dataclass
class Step:
    tool: str
    arguments: dict
    result: str
    error: str | None
    seconds: float


@dataclass
class Result:
    question: str
    answer: str
    trace: list[Step] = field(default_factory=list)
    turns: int = 0
    seconds: float = 0.0
    finished: bool = True

    @property
    def blocked(self) -> int:
        return sum(1 for s in self.trace if s.error)


class Copilot:
    def __init__(self, provider: Provider, max_turns: int = 8, server_cmd: list[str] | None = None,
                 env: dict | None = None):
        self.provider = provider
        self.max_turns = max_turns
        self.server_cmd = server_cmd or [sys.executable, "-m", "copilot.server"]
        # The MCP stdio client passes only a minimal environment to the server by
        # default, so forward ours: LAKEHOUSE_DB and friends must reach the server.
        self.env = {**os.environ, **(env or {})}
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[dict] = []

    async def __aenter__(self) -> "Copilot":
        params = StdioServerParameters(command=self.server_cmd[0], args=self.server_cmd[1:], env=self.env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        listed = await self.session.list_tools()
        self.tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                      for t in listed.tools]
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def _call(self, name: str, arguments: dict) -> tuple[str, str | None]:
        res = await self.session.call_tool(name, arguments)
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        error = None
        if res.isError:
            error = text[:300]
        else:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "error" in parsed:
                    error = str(parsed["error"])
            except ValueError:
                pass
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "\n...[truncated]"
        return text, error

    async def ask(self, question: str) -> Result:
        started = time.monotonic()
        messages: list[dict] = [{"role": "user", "content": question}]
        result = Result(question=question, answer="")
        for turn_no in range(1, self.max_turns + 1):
            turn: Turn = self.provider.complete(SYSTEM_PROMPT, messages, self.tools)
            result.turns = turn_no
            if not turn.tool_calls:
                result.answer = (turn.text or "").strip()
                break
            messages.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls})
            for call in turn.tool_calls:
                t0 = time.monotonic()
                text, error = await self._call(call.name, call.arguments)
                result.trace.append(Step(call.name, call.arguments, text, error, time.monotonic() - t0))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": text})
        else:
            result.finished = False
            result.answer = "I could not finish within the allowed number of steps."
        result.seconds = time.monotonic() - started
        return result
