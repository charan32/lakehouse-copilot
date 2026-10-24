"""SQL guardrails: the agent may only read, and only from the modeled layers.

Three independent layers keep the agent from changing or leaking anything:

1. This module parses the SQL with sqlglot and rejects anything that is not a
   single read-only query over allowed schemas, or that calls a function able to
   touch files, the network or the environment.
2. The DuckDB connection is opened read-only with external access disabled
   (see ``warehouse.py``), so even a query that slipped past the parser cannot
   write or read files.
3. Every query gets an enforced row limit and a time limit.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

ALLOWED_SCHEMAS = {"gold", "silver", "monitoring", "reference"}
MAX_ROWS = 200

# Statement types that change data or the database.
WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Copy, exp.Set, exp.Pragma, exp.Use, exp.Attach, exp.Detach,
    exp.TruncateTable, exp.Transaction, exp.Commit, exp.Rollback,
)
READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)

# Functions that can read files, reach the network or reveal the environment.
BLOCKED_FUNCTION_PREFIXES = ("read_", "parquet_", "glob", "getenv", "sniff_", "iceberg_", "delta_scan")
BLOCKED_FUNCTIONS = {"current_setting", "duckdb_settings", "duckdb_secrets", "load_extension", "query", "query_table"}


class GuardError(ValueError):
    """The query is not allowed. The message says why, in words the agent can act on."""


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() or "").lower()


def check(sql: str) -> exp.Expression:
    """Parse ``sql`` and return its syntax tree if it is a safe read-only query."""
    if not sql or not sql.strip():
        raise GuardError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise GuardError(f"Could not parse the SQL: {str(exc).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise GuardError("Send exactly one statement per call.")
    tree = statements[0]

    if tree.find(*WRITE_NODES) is not None or not isinstance(tree, READ_ROOTS):
        raise GuardError("Only read-only SELECT queries are allowed. This lakehouse is read-only for the copilot.")

    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            raise GuardError("Table functions are not allowed. Query a table in the gold, silver, monitoring or reference schema.")
        name = table.name.lower()
        schema = (table.db or "").lower()
        if not schema and name in ctes:
            continue
        if table.catalog:
            raise GuardError("Do not reference other databases.")
        if schema not in ALLOWED_SCHEMAS:
            where = f"{schema}.{name}" if schema else name
            raise GuardError(
                f"Table {where!r} is not available. Use schema-qualified tables from: "
                + ", ".join(sorted(ALLOWED_SCHEMAS)) + " (call list_tables to see them)."
            )

    for fn in tree.find_all(exp.Func):
        name = _function_name(fn)
        if name in BLOCKED_FUNCTIONS or name.startswith(BLOCKED_FUNCTION_PREFIXES):
            raise GuardError(f"Function {name!r} is not allowed: it can reach files, settings or the network.")
    for lit in tree.find_all(exp.Literal):
        text = str(lit.this).lower()
        if lit.is_string and ("://" in text or text.startswith(("/", "~", "../"))):
            raise GuardError("File paths and URLs are not allowed in queries.")
    return tree


def limited(sql: str, limit: int) -> str:
    """Return the checked query wrapped with a row limit no larger than ``MAX_ROWS``."""
    tree = check(sql)
    n = max(1, min(int(limit), MAX_ROWS))
    return f"SELECT * FROM ({tree.sql(dialect='duckdb')}) AS q LIMIT {n}"
