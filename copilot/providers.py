"""LLM providers behind one small interface.

The agent keeps a provider-neutral message history; each provider converts it to
its own API format. Supported:

* ``ollama``    - any local model with tool calling (default ``qwen2.5:7b``), via
                  Ollama's OpenAI-compatible endpoint. Free, no API key.
* ``openai``    - OpenAI or any OpenAI-compatible server (``OPENAI_API_KEY``).
* ``anthropic`` - Claude models (``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class Provider:
    name = "base"
    model = ""

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:  # pragma: no cover
        raise NotImplementedError


# ---------------- OpenAI-compatible (OpenAI, Ollama, vLLM, ...) ----------------

def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in m["tool_calls"]
                ]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class OpenAICompatible(Provider):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, name: str = "openai"):
        from openai import OpenAI

        self.name, self.model = name, model
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY") or "none")

    def complete(self, system, messages, tools):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=to_openai_messages(system, messages),
            tools=to_openai_tools(tools),
            temperature=0,
        )
        msg = resp.choices[0].message
        calls = []
        for i, c in enumerate(msg.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": c.function.arguments}
            calls.append(ToolCall(c.id or f"call_{i}", c.function.name, args))
        return Turn(msg.content, calls)


# ---------------- Anthropic ----------------

def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            # Consecutive tool results belong in one user message.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return out


class Anthropic(Provider):
    def __init__(self, model: str):
        import anthropic

        self.name, self.model = "anthropic", model
        self.client = anthropic.Anthropic()

    def complete(self, system, messages, tools):
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=1500,
            temperature=0,
            messages=to_anthropic_messages(messages),
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                   for t in tools],
        )
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return Turn(text, calls)


def make_provider(name: str, model: str | None = None) -> Provider:
    name = name.lower()
    if name == "ollama":
        return OpenAICompatible(
            model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            name="ollama",
        )
    if name == "openai":
        return OpenAICompatible(model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                base_url=os.getenv("OPENAI_BASE_URL"), name="openai")
    if name == "anthropic":
        return Anthropic(model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    raise ValueError(f"Unknown provider {name!r}. Use ollama, openai or anthropic.")
