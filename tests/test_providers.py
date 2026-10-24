"""Exercise the real OpenAI and Anthropic SDK code paths against local fake servers."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from copilot.agent import Copilot
from copilot.providers import Anthropic, OpenAICompatible, ToolCall, to_anthropic_messages

pytestmark = pytest.mark.asyncio


class FakeLLM:
    """Replies with a tool call first, then a final answer quoting the tool result."""

    def __init__(self, style):
        self.style, self.requests = style, []

    def reply(self, body):
        self.requests.append(body)
        msgs = body["messages"]
        tool_result = None
        for m in msgs:
            if m.get("role") == "tool":
                tool_result = m["content"]
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for b in m["content"]:
                    if b.get("type") == "tool_result":
                        tool_result = b["content"]
        if self.style == "openai":
            if tool_result is None:
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "get_metric", "arguments": json.dumps({"name": "orders"})}}]}
            else:
                value = json.loads(tool_result)["rows"][0][0]
                msg = {"role": "assistant", "content": f"There were {value} orders."}
            return {"id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]}
        if tool_result is None:
            content = [{"type": "tool_use", "id": "t1", "name": "get_metric", "input": {"name": "orders"}}]
            stop = "tool_use"
        else:
            value = json.loads(tool_result)["rows"][0][0]
            content, stop = [{"type": "text", "text": f"There were {value} orders."}], "end_turn"
        return {"id": "m", "type": "message", "role": "assistant", "model": "fake", "content": content,
                "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


def serve(fake):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps(fake.reply(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def test_openai_compatible_provider_runs_tool_loop(wh):
    fake = FakeLLM("openai")
    httpd = serve(fake)
    provider = OpenAICompatible("fake", base_url=f"http://127.0.0.1:{httpd.server_port}/v1", api_key="x")
    async with Copilot(provider) as c:
        result = await c.ask("How many orders?")
    httpd.shutdown()
    expected = wh.get_metric("orders")["rows"][0][0]
    assert result.answer == f"There were {expected} orders."
    assert [s.tool for s in result.trace] == ["get_metric"]
    second = fake.requests[1]["messages"]
    assert second[0]["role"] == "system" and second[-1]["role"] == "tool"
    assert {t["function"]["name"] for t in fake.requests[0]["tools"]} >= {"run_sql", "get_metric"}


async def test_anthropic_provider_runs_tool_loop(wh, monkeypatch):
    fake = FakeLLM("anthropic")
    httpd = serve(fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{httpd.server_port}")
    provider = Anthropic("fake")
    async with Copilot(provider) as c:
        result = await c.ask("How many orders?")
    httpd.shutdown()
    assert result.answer == f"There were {wh.get_metric('orders')['rows'][0][0]} orders."
    assert fake.requests[0]["system"].startswith("You are Lakehouse Copilot")


def test_consecutive_tool_results_share_one_user_message():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [ToolCall("a", "x", {}), ToolCall("b", "y", {})]},
        {"role": "tool", "tool_call_id": "a", "name": "x", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "name": "y", "content": "2"},
    ]
    out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["a", "b"]
