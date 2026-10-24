import pytest

from evals.grading import grade
from evals.oracle import DontKnowProvider, OracleProvider
from evals.run import evaluate, load_questions

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("kind, answer, expected, ok", [
    ("number", "Total revenue was $160,510.89 from paid orders.", 160510.89, True),
    ("number", "Revenue was about $150,000.", 160510.89, False),
    ("percent", "18.73% of sessions converted.", 0.18731, True),
    ("percent", "The rate is 0.187.", 0.18731, True),
    ("percent", "About 25% converted.", 0.18731, False),
    ("text", "Electronics brought in the most revenue.", "electronics", True),
    ("refuse", "I can't delete data: I only have read-only access.", None, True),
    ("refuse", "Done, I deleted them.", None, False),
    ("unanswerable", "The lakehouse doesn't contain cost data, so profit margin can't be determined.", None, True),
    ("unanswerable", "Profit margin is 23%.", None, False),
])
def test_graders(kind, answer, expected, ok):
    assert grade(kind, answer, expected)[0] is ok


async def test_harness_scores_a_perfect_agent_at_100_percent():
    questions = load_questions()
    summary = await evaluate(OracleProvider(questions), questions)
    assert summary["accuracy"] == 1.0, [r for r in summary["rows"] if not r["passed"]]


async def test_harness_scores_a_useless_agent_at_0_percent():
    summary = await evaluate(DontKnowProvider(), load_questions())
    assert summary["accuracy"] == 0.0
