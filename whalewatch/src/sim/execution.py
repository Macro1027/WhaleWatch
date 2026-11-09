"""
Execution strategies and fill model.

Both strategies take the same `MarketReplay` and the same `ExecutionConfig`
(target_qty, slice_count, deadline, side). They differ only in when they fire.

Fill model (MVP, explicit):
- Each strategy emits MARKETABLE child orders that cross the spread.
- A fill occurs at `ask_price(mid[i]) = mid[i] * (1 + half_spread_bps/1e4)`.
- No resting limit orders, no queue position, no market impact beyond spread.
- Resting-fill-on-touch is NOT modelled; we avoid the "mid touched my limit
  so I filled" fallacy.
- Terminal pending quantity at deadline is reported as unfilled inventory;
  the evaluator charges it at close mid as opportunity cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sim.book import MarketReplay


@dataclass(frozen=True)
class Fill:
    t_s: float
    qty: float
    price: float
    strategy_id: str


@dataclass(frozen=True)
class ExecutionConfig:
    target_qty: float = 2.0
    slice_count: int = 20
    deadline_s: float = 600.0
    side: str = "buy"  # MVP supports buy only; sell is a stretch
    half_spread_bps: float = 1.0

    @property
    def slice_qty(self) -> float:
        return self.target_qty / self.slice_count

    @property
    def slice_interval_s(self) -> float:
        return self.deadline_s / self.slice_count


@dataclass(frozen=True)
class GatedConfig:
    """Hysteresis config for the toxicity-gated strategy."""
    threshold: float = 7.0            # absolute 1s signed trade imbalance
    cooldown_steps: int = 50          # must be below threshold this many steps


def _ask(mid: float, half_spread_bps: float) -> float:
    return mid * (1 + half_spread_bps / 10_000)


def baseline_twap(
    replay: MarketReplay,
    cfg: ExecutionConfig,
) -> tuple[list[Fill], float]:
    """Fire one `slice_qty` child at each of `slice_count` evenly spaced times."""
    if cfg.side != "buy":
        raise NotImplementedError("MVP: buy side only")
    fills: list[Fill] = []
    n = replay.n_steps
    for k in range(cfg.slice_count):
        target_t = k * cfg.slice_interval_s
        idx = min(int(round(target_t / replay.dt_s)), n - 1)
        fills.append(Fill(
            t_s=float(replay.t_s[idx]),
            qty=cfg.slice_qty,
            price=_ask(float(replay.mid[idx]), cfg.half_spread_bps),
            strategy_id="baseline_twap",
        ))
    return fills, 0.0


def gated_twap(
    replay: MarketReplay,
    signal: np.ndarray,
    cfg: ExecutionConfig,
    gated_cfg: GatedConfig,
) -> tuple[list[Fill], float]:
    """
    Same slice schedule as baseline_twap, but at each step defer the scheduled
    slice into a pending backlog unless the signal has been <= threshold for
    `cooldown_steps` consecutive steps. The entire backlog fires at once when
    that condition is met.

    Fairness: identical replay, identical target quantity, identical deadline,
    identical slice boundaries, identical fill price model.
    """
    if cfg.side != "buy":
        raise NotImplementedError("MVP: buy side only")
    if signal.shape[0] != replay.n_steps:
        raise ValueError("signal length must match replay length")
    fills: list[Fill] = []
    pending = 0.0
    next_slice = 0
    steps_below = 0
    for i in range(replay.n_steps):
        if signal[i] <= gated_cfg.threshold:
            steps_below += 1
        else:
            steps_below = 0
        if (
            next_slice < cfg.slice_count
            and replay.t_s[i] >= next_slice * cfg.slice_interval_s - 1e-9
        ):
            pending += cfg.slice_qty
            next_slice += 1
        if pending > 0 and steps_below >= gated_cfg.cooldown_steps:
            fills.append(Fill(
                t_s=float(replay.t_s[i]),
                qty=pending,
                price=_ask(float(replay.mid[i]), cfg.half_spread_bps),
                strategy_id="gated_twap",
            ))
            pending = 0.0
    return fills, pending
