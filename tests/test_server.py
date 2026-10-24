import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.asyncio


async def test_mcp_server_over_stdio():
    params = StdioServerParameters(command=sys.executable, args=["-m", "copilot.server"])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = {t.name for t in (await s.list_tools()).tools}
        assert names == {"list_tables", "describe_table", "list_metrics", "get_metric",
                         "run_sql", "pipeline_health", "contract_violations"}

        res = await s.call_tool("get_metric", {"name": "orders"})
        assert json.loads(res.content[0].text)["rows"][0][0] > 0

        res = await s.call_tool("run_sql", {"sql": "drop table gold.fct_orders"})
        assert "read-only" in json.loads(res.content[0].text)["error"]
