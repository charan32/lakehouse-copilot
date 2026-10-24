"""Read-only access to the lakehouse's DuckDB file (the silver, gold and monitoring layers dbt builds).

Every function here returns plain JSON-serializable data, so it can be handed
straight to an MCP client or an LLM.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import threading
from functools import cached_property
from pathlib import Path

import duckdb
import yaml

from . import guard

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = str(REPO_ROOT / ".lakehouse" / "streaming-lakehouse" / "dbt" / "target" / "lakehouse.duckdb")
QUERY_TIMEOUT_S = float(os.getenv("COPILOT_QUERY_TIMEOUT", "10"))
METRICS_FILE = Path(__file__).with_name("metrics.yml")


def _jsonable(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return v.total_seconds()
    return v


class Warehouse:
    def __init__(self, db_path: str | None = None, manifest_path: str | None = None):
        self.db_path = db_path or os.getenv("LAKEHOUSE_DB", DEFAULT_DB)
        if not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"Lakehouse database not found at {self.db_path}. Build it with `make demo-data` "
                "or point LAKEHOUSE_DB at the lakehouse.duckdb file."
            )
        self.manifest_path = manifest_path or os.getenv(
            "LAKEHOUSE_MANIFEST", str(Path(self.db_path).with_name("manifest.json"))
        )
        # Defense in depth: read-only, and no access to files or the network from SQL.
        self.con = duckdb.connect(self.db_path, read_only=True, config={"enable_external_access": False})
        self._lock = threading.Lock()

    # ---------- low level ----------

    def _run(self, sql: str, params: list | None = None) -> dict:
        timer = threading.Timer(QUERY_TIMEOUT_S, self.con.interrupt)
        with self._lock:
            timer.start()
            try:
                cur = self.con.execute(sql, params or [])
                cols = [d[0] for d in cur.description]
                rows = [[_jsonable(v) for v in r] for r in cur.fetchall()]
            except duckdb.InterruptException as exc:
                raise guard.GuardError(f"Query took longer than {QUERY_TIMEOUT_S:.0f}s and was stopped.") from exc
            finally:
                timer.cancel()
        return {"columns": cols, "rows": rows, "row_count": len(rows)}

    # ---------- catalog ----------

    @cached_property
    def _descriptions(self) -> dict:
        """Model and column descriptions from the dbt manifest, keyed by schema.table."""
        try:
            manifest = json.loads(Path(self.manifest_path).read_text())
        except (OSError, ValueError):
            return {}
        out = {}
        for node in manifest.get("nodes", {}).values():
            if node.get("resource_type") in ("model", "seed"):
                out[f"{node['schema']}.{node['name']}"] = {
                    "description": node.get("description", ""),
                    "columns": {c: v.get("description", "") for c, v in node.get("columns", {}).items()},
                }
        return out

    def list_tables(self) -> list[dict]:
        res = self._run(
            "select table_schema, table_name from information_schema.tables "
            "where table_schema in (select unnest(?)) order by 1, 2",
            [sorted(guard.ALLOWED_SCHEMAS)],
        )
        return [
            {"table": f"{s}.{t}", "description": self._descriptions.get(f"{s}.{t}", {}).get("description", "")}
            for s, t in res["rows"]
        ]

    def describe_table(self, table: str) -> dict:
        schema, _, name = table.lower().partition(".")
        if not name or schema not in guard.ALLOWED_SCHEMAS:
            raise guard.GuardError("Use a schema-qualified table name such as gold.fct_orders (see list_tables).")
        cols = self._run(
            "select column_name, data_type from information_schema.columns "
            "where table_schema = ? and table_name = ? order by ordinal_position",
            [schema, name],
        )["rows"]
        if not cols:
            raise guard.GuardError(f"Table {table!r} does not exist. Call list_tables to see what is available.")
        docs = self._descriptions.get(f"{schema}.{name}", {})
        count = self._run(f'select count(*) from "{schema}"."{name}"')["rows"][0][0]
        sample = self._run(f'select * from "{schema}"."{name}" limit 3')
        return {
            "table": f"{schema}.{name}",
            "description": docs.get("description", ""),
            "row_count": count,
            "columns": [{"name": c, "type": t, "description": docs.get("columns", {}).get(c, "")} for c, t in cols],
            "sample_rows": sample,
        }

    # ---------- queries ----------

    def run_sql(self, sql: str, limit: int = 50) -> dict:
        return self._run(guard.limited(sql, limit))

    # ---------- semantic layer ----------

    @cached_property
    def metrics(self) -> dict:
        return yaml.safe_load(METRICS_FILE.read_text())

    def list_metrics(self) -> list[dict]:
        return [
            {"name": k, "description": v["description"], "dimensions": v.get("dimensions", []),
             "filterable_by_date": bool(v.get("date_column"))}
            for k, v in self.metrics.items()
        ]

    def get_metric(self, name: str, start_date: str | None = None, end_date: str | None = None,
                   group_by: str | None = None) -> dict:
        m = self.metrics.get(name)
        if m is None:
            raise guard.GuardError(f"Unknown metric {name!r}. Available: {', '.join(self.metrics)}.")
        where, params = [], []
        for value, op in ((start_date, ">="), (end_date, "<=")):
            if value:
                if not m.get("date_column"):
                    raise guard.GuardError(f"Metric {name!r} has no date column, so it cannot be filtered by date.")
                try:
                    dt.date.fromisoformat(value)
                except ValueError as exc:
                    raise guard.GuardError(f"Dates must look like 2026-01-31, got {value!r}.") from exc
                where.append(f"{m['date_column']} {op} cast(? as date)")
                params.append(value)
        dims = []
        if group_by:
            allowed = list(m.get("dimensions", [])) + ([m["date_column"]] if m.get("date_column") else [])
            if group_by not in allowed:
                raise guard.GuardError(f"Metric {name!r} can be grouped by: {', '.join(allowed) or 'nothing'}.")
            dims = [group_by]
        select = ", ".join(dims + [f"{m['expression']} as {name}"])
        sql = f"select {select} from {m['table']}"
        if where:
            sql += " where " + " and ".join(where)
        if dims:
            sql += f" group by {group_by} order by {name} desc nulls last"
        result = self._run(sql, params)
        result.update({"metric": name, "definition": m["description"], "sql": sql})
        return result

    # ---------- pipeline operations ----------

    def pipeline_health(self) -> dict:
        return self._run("select * from monitoring.pipeline_health order by day")

    def contract_violations(self, event_type: str | None = None, limit: int = 20) -> dict:
        sql = ("select event_type, failed_field, sum(rejected_events) as rejected_events, "
               "any_value(example_error) as example_error from monitoring.contract_rejections")
        params = []
        if event_type:
            sql += " where event_type = ?"
            params.append(event_type)
        sql += f" group by all order by rejected_events desc limit {max(1, min(int(limit), 100))}"
        return self._run(sql, params)
