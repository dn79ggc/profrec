"""Starter requests shown as one-click chips under the chat box.

Each must parse into at least one criterion with the offline RuleBasedParser (tests/test_app_smoke.py),
so the chips work with or without a Gemini key.
"""
EXAMPLE_REQUESTS: tuple[str, ...] = (
    "Easy-going and helpful",
    "Clear expectations and light workload",
    "Challenging but fair, no Friday classes",
    "Engaging lectures, no mornings",
)
