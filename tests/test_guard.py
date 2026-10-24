import pytest

from copilot import guard

SAFE = [
    "select sum(order_total) from gold.fct_orders where is_paid",
    "with t as (select * from gold.fct_orders) select count(*) from t",
    "select category, sum(line_total) from gold.fct_order_items group by 1 order by 2 desc",
    "select * from monitoring.pipeline_health",
    "select o.order_id from gold.fct_orders o join silver.stg_payments p using (order_id)",
]

UNSAFE = [
    ("delete from gold.fct_orders", "read-only"),
    ("drop table gold.fct_orders", "read-only"),
    ("update gold.fct_orders set order_total = 0", "read-only"),
    ("insert into gold.fct_orders select * from gold.fct_orders", "read-only"),
    ("create table gold.vip as select 1", "read-only"),
    ("copy gold.fct_orders to 'out.csv'", "read-only"),
    ("attach 'other.db'", "read-only"),
    ("install httpfs", "read-only"),
    ("pragma database_list", "read-only"),
    ("set threads = 1", "read-only"),
    ("select 1; drop table gold.fct_orders", "exactly one"),
    ("select * from read_csv('/etc/passwd')", "Table functions"),
    ("select * from gold.fct_orders, read_parquet('s3://bucket/x.parquet')", "Table functions"),
    ("select * from 'data.parquet'", "not available"),
    ("select * from bronze.events", "not available"),
    ("select * from fct_orders", "not available"),
    ("select * from information_schema.tables", "not available"),
    ("select * from other_db.gold.fct_orders", "other databases"),
    ("select getenv('HOME')", "not allowed"),
    ("select current_setting('home_directory')", "not allowed"),
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    guard.check(sql)


@pytest.mark.parametrize("sql, reason", UNSAFE)
def test_unsafe_queries_are_blocked_with_a_reason(sql, reason):
    with pytest.raises(guard.GuardError, match=reason):
        guard.check(sql)


def test_limit_is_capped():
    assert guard.limited("select * from gold.fct_orders", 10_000).endswith(f"LIMIT {guard.MAX_ROWS}")
    assert guard.limited("select * from gold.fct_orders", 5).endswith("LIMIT 5")
