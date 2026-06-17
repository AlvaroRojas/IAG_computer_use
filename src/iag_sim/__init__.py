"""Murex accounting-simulation before/after comparator.

Drives the Murex web UI with OpenAI computer-use (Playwright harness),
runs the accounting simulation per trade across a "before" and "after"
environment, exports CSV, then diffs the aggregates deterministically.
"""

__version__ = "0.1.0"
