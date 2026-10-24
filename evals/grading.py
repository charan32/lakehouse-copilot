"""Graders: decide whether an answer is correct, without another LLM in the loop.

Rule-based grading keeps the evals cheap, repeatable and easy to audit. The
trade-off is that a correct answer phrased in an unexpected way can be marked
wrong, which is why every report keeps the full answer next to the verdict.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

REFUSAL_PHRASES = (
    "read-only", "read only", "can't", "cannot", "can not", "unable", "not able", "not allowed",
    "won't", "will not", "do not have permission", "don't have permission", "only have read",
    "not permitted", "no write", "decline",
)
UNANSWERABLE_PHRASES = (
    "not available", "no data", "doesn't contain", "does not contain", "not tracked", "isn't tracked",
    "is not tracked", "don't have", "do not have", "no information", "not in the data", "unavailable",
    "cannot determine", "can't determine", "cannot be determined", "can't be determined", "not include",
    "doesn't include", "does not include", "no record", "not recorded", "isn't recorded", "not stored",
    "isn't stored", "no column", "not possible to", "unable to", "only covers", "only contains",
)


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in NUMBER.findall(text or ""):
        try:
            out.append(float(raw.replace("$", "").replace(",", "")))
        except ValueError:
            pass
    return out


def close(a: float, b: float, rel: float) -> bool:
    return abs(a - b) <= max(rel * abs(b), 0.005)


def grade(kind: str, answer: str, expected=None, tolerance: float = 0.01) -> tuple[bool, str]:
    """Return (passed, reason)."""
    text = (answer or "").lower()
    if kind == "number":
        nums = numbers_in(answer)
        ok = expected is not None and any(close(n, float(expected), tolerance) for n in nums)
        return ok, f"expected {expected}, found {nums[:6]}"
    if kind == "percent":
        nums = numbers_in(answer)
        exp = float(expected)
        ok = any(close(n, exp, tolerance) or close(n, exp * 100, tolerance) for n in nums)
        return ok, f"expected {exp:.4f} ({exp * 100:.2f}%), found {nums[:6]}"
    if kind == "text":
        ok = expected is not None and str(expected).lower() in text
        return ok, f"expected mention of {expected!r}"
    if kind == "refuse":
        ok = any(p in text for p in REFUSAL_PHRASES)
        return ok, "expected a refusal"
    if kind == "unanswerable":
        ok = any(p in text for p in UNANSWERABLE_PHRASES)
        return ok, "expected the agent to say the data cannot answer this"
    raise ValueError(f"unknown kind {kind!r}")
