"""Test doubles for the eval harness.

``OracleProvider`` plays the role of a perfect model: for each question it calls
the real MCP ``run_sql`` tool with the question's reference SQL and answers with
the result. It never sees the expected answer directly, so a 100% score proves
the harness, the MCP round trip and the graders work end to end. It says
nothing about how good a real model is.

``DontKnowProvider`` answers "I don't know" to everything and must score 0 on
answerable questions, which proves the graders are not too lenient.
"""

from __future__ import annotations

import json

from copilot.providers import Provider, ToolCall, Turn


class OracleProvider(Provider):
    name, model = "oracle", "reference-sql"

    def __init__(self, questions: list[dict]):
        self.by_text = {q["question"]: q for q in questions}

    def complete(self, system, messages, tools):
        q = self.by_text[messages[0]["content"]]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if q["kind"] == "refuse":
            return Turn("I can't do that: I only have read-only access to the lakehouse.")
        if q["kind"] == "unanswerable":
            return Turn("The lakehouse does not contain that information, so I cannot answer it.")
        if not tool_msgs:
            return Turn(None, [ToolCall("call_1", "run_sql", {"sql": q["reference_sql"], "limit": 5})])
        value = json.loads(tool_msgs[-1]["content"])["rows"][0][0]
        if q["kind"] == "percent":
            return Turn(f"It is {value * 100:.2f}% (from {q['category']} data).")
        return Turn(f"The answer is {value}.")


class DontKnowProvider(Provider):
    name, model = "dont-know", "none"

    def complete(self, system, messages, tools):
        return Turn("I'm not sure.")
