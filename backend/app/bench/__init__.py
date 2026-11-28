"""Accuracy benchmark: reconstruct Google Scanned Objects from one rendered view
and compare each method's mesh with the scanned ground truth.

    python -m app.bench run --out bench-results          # needs a GPU and internet
    python -m app.bench report --out bench-results

See docs/benchmark.md for the protocol and the reasoning behind it.
"""
