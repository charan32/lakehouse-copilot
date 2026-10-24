import duckdb
import pytest

from copilot.guard import GuardError


def test_lists_and_describes_modeled_tables(wh):
    tables = {t["table"] for t in wh.list_tables()}
    assert {"gold.fct_orders", "gold.dim_customers", "monitoring.pipeline_health"} <= tables
    assert not any(t.startswith("bronze.") for t in tables)
    desc = wh.describe_table("gold.fct_orders")
    assert desc["row_count"] > 0 and desc["description"]
    assert "order_total" in [c["name"] for c in desc["columns"]]


def test_metrics_match_hand_written_sql(wh):
    revenue = wh.get_metric("revenue")["rows"][0][0]
    direct = wh._run("select sum(order_total) from gold.fct_orders where is_paid")["rows"][0][0]
    assert revenue == pytest.approx(direct)
    by_cat = wh.get_metric("revenue", group_by="category")["rows"]
    assert sum(v for _, v in by_cat) == pytest.approx(revenue)


def test_metric_errors_are_actionable(wh):
    with pytest.raises(GuardError, match="Available"):
        wh.get_metric("profit")
    with pytest.raises(GuardError, match="grouped by"):
        wh.get_metric("revenue", group_by="user_id")
    with pytest.raises(GuardError, match="YYYY|2026-01-31"):
        wh.get_metric("revenue", start_date="last week")


def test_database_itself_refuses_writes_and_files(wh):
    # Second line of defense, independent of the SQL parser.
    with pytest.raises(duckdb.Error):
        wh.con.execute("delete from gold.fct_orders")
    with pytest.raises(duckdb.Error):
        wh.con.execute("select * from read_csv('/etc/passwd')")
