"""
Toxicity signal: raw signed trade imbalance over the trailing 1 second.

Causal only. MUST NOT read `is_synthetic_event` or any future-timestamp data.
Phase-0 experiment validated that this signal (as absolute threshold with
~5s sustained-below hysteresis in the gating logic) reliably separates the
scripted whale from base flow across a wide threshold band (5..15), with
mean saving ~3 bps on a 2-BTC / 10-min parent order.

A z-scored variant was tried first; it collapses to ~0 when the
normalisation window is shorter than the toxic regime. Absolute threshold
on raw imbalance is more honest.
"""
from __future__ import annotations

import numpy as np


def trade_imbalance_1s(
    buy_trades: np.ndarray,
    sell_trades: np.ndarray,
    window_steps: int,
) -> np.ndarray:
    """
    Trailing 1-second signed trade imbalance.
    `window_steps` is the number of steps covering 1s at the replay's dt.
    Element i depends ONLY on indices in [max(0, i-window+1), i].
    """
    signed = buy_trades - sell_trades
    cs = np.concatenate([[0.0], np.cumsum(signed)])
    out = np.zeros_like(signed, dtype=float)
    for i in range(len(signed)):
        lo = max(0, i - window_steps + 1)
        out[i] = cs[i + 1] - cs[lo]
    return out
