"""MCP server exposing the lakehouse to AI agents through read-only tools.

Run it over stdio (the default for MCP clients such as Claude Desktop, Cursor or
this repo's agent):

    python -m copilot.server

Every tool returns JSON. When a request is not allowed or is malformed the tool
returns ``{"error": "..."}`` with a sentence the model can act on, instead of
failing the whole conversation.
"""

from __future__ import annotations

import functools
import logging
import sys

from mcp.server.fastmcp import FastMCP

from .guard import GuardError
from .warehouse import Warehouse

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("copilot.server")

mcp = FastMCP(
    "lakehouse-copilot",
    instructions=(
        "Read-only access to an e-commerce lakehouse (gold business tables, silver cleaned events, "
        "monitoring tables about data quality). Prefer get_metric for business numbers; use "
        "describe_table before writing SQL; run_sql accepts one read-only SELECT."
    ),
)

_wh: Warehouse | None = None


def wh() -> Warehouse:
    global _wh
    if _wh is None:
        _wh = Warehouse()
    return _wh


def safe(fn):
    """Turn guardrail and database errors into a readable ``{"error": ...}`` result."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except GuardError as exc:
            log.info("blocked %s: %s", fn.__name__, exc)
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - surface database errors to the model
            log.info("failed %s: %s", fn.__name__, exc)
            return {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
    return wrapper


@mcp.tool()
@safe
def list_tables() -> list[dict] | dict:
    """List the tables the copilot can read, with a one-line description of each."""
    return wh().list_tables()


@mcp.tool()
@safe
def describe_table(table: str) -> dict:
    """Show a table's columns, types, descriptions, row count and three sample rows.

    Args:
        table: schema-qualified name, for example "gold.fct_orders".
    """
    return wh().describe_table(table)


@mcp.tool()
@safe
def list_metrics() -> list[dict] | dict:
    """List the business metrics with agreed definitions (revenue, conversion_rate, ...)."""
    return wh().list_metrics()


@mcp.tool()
@safe
def get_metric(name: str, start_date: str | None = None, end_date: str | None = None,
               group_by: str | None = None) -> dict:
    """Compute a business metric. Prefer this over writing SQL for business numbers.

    Args:
        name: metric name from list_metrics, for example "revenue".
        start_date: optional inclusive start date, YYYY-MM-DD.
        end_date: optional inclusive end date, YYYY-MM-DD.
        group_by: optional dimension from list_metrics, or the metric's date column.
    """
    return wh().get_metric(name, start_date, end_date, group_by)


@mcp.tool()
@safe
def run_sql(sql: str, limit: int = 50) -> dict:
    """Run one read-only SELECT against the gold, silver, monitoring or reference schemas.

    Writes, file access and other databases are blocked. At most 200 rows are returned.

    Args:
        sql: a single SELECT statement using schema-qualified table names.
        limit: maximum rows to return (1 to 200).
    """
    return wh().run_sql(sql, limit)


@mcp.tool()
@safe
def pipeline_health() -> dict:
    """Per-day ingestion health: rows, duplicates removed, reject rate, freshness and late events."""
    return wh().pipeline_health()


@mcp.tool()
@safe
def contract_violations(event_type: str | None = None, limit: int = 20) -> dict:
    """Which producers broke the data contracts, on which field, how often, with an example error.

    Args:
        event_type: optional filter, for example "page_view".
        limit: maximum rows (1 to 100).
    """
    return wh().contract_violations(event_type, limit)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
