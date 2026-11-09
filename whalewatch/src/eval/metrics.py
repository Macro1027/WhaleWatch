"""
Side-aware implementation shortfall and markout metrics.

Conventions (BUY side):
- IS is positive when execution is MORE expensive than the arrival-mid
  benchmark (i.e. a cost).
- Unfilled inventory is charged at close mid as opportunity cost.
- Taker fee is applied pro-rata to the filled portion of the target.

500ms markout (BUY): +bps if mid has risen 500ms after the fill
(adverse-selection cost), -bps if mid has fallen (favourable mark).
A positive markout implies the buyer was picked off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sim.book import MarketReplay
from src.sim.execution import Fill


@dataclass(frozen=True)
class ExecutionResult:
    strategy_id: str
    is_bps: float
    filled_qty: float
    unfilled_qty: float
    fill_count: int
    avg_markout_500ms_bps: float
    vwap_price: float
    arrival_mid: float
    close_mid: float


def side_aware_is_bps(
    fills: list[Fill],
    unfilled_qty: float,
    arrival_mid: float,
    close_mid: float,
    target_qty: float,
    fee_bps: float = 1.0,
) -> tuple[float, float]:
    """
    Returns (is_bps, filled_qty).

    IS (buy) = 10_000 * (sum(fill_px * fill_qty) + close_mid * unfilled_qty
                        - arrival_mid * target_qty)
                       / (arrival_mid * target_qty)
             + fee_bps * (filled_qty / target_qty)
    """
    filled_qty = sum(f.qty for f in fills)
    filled_cost = sum(f.price * f.qty for f in fills)
    unfilled_cost = close_mid * unfilled_qty
    excess = filled_cost + unfilled_cost - arrival_mid * target_qty
    is_bps = 10_000 * excess / (arrival_mid * target_qty)
    is_bps += fee_bps * (filled_qty / target_qty)
    return is_bps, filled_qty


def markout_500ms_bps(
    fills: list[Fill],
    replay: MarketReplay,
    side: str = "buy",
) -> float:
    """
    Signed 500ms markout in bps, averaged qty-weighted across fills.
    BUY convention: positive = adverse (mid rose after fill).
    """
    if not fills:
        return 0.0
    if side != "buy":
        raise NotImplementedError("MVP: buy side only")
    horizon_steps = max(1, int(round(0.5 / replay.dt_s)))
    n = replay.n_steps
    total_qty = sum(f.qty for f in fills)
    weighted_sum = 0.0
    for f in fills:
        fill_idx = min(int(round(f.t_s / replay.dt_s)), n - 1)
        future_idx = min(fill_idx + horizon_steps, n - 1)
        mid_at_fill = float(replay.mid[fill_idx])
        mid_future = float(replay.mid[future_idx])
        # buy: +bps if price rose = adverse
        bps = 10_000 * (mid_future - mid_at_fill) / mid_at_fill
        weighted_sum += bps * f.qty
    return weighted_sum / total_qty


def evaluate(
    fills: list[Fill],
    unfilled_qty: float,
    replay: MarketReplay,
    target_qty: float,
    fee_bps: float = 1.0,
    side: str = "buy",
) -> ExecutionResult:
    """Package IS, markout and descriptive stats for one strategy run."""
    is_bps, filled_qty = side_aware_is_bps(
        fills, unfilled_qty, replay.arrival_mid, replay.close_mid,
        target_qty, fee_bps,
    )
    mo = markout_500ms_bps(fills, replay, side)
    vwap = (
        sum(f.price * f.qty for f in fills) / filled_qty
        if filled_qty > 0 else float("nan")
    )
    strategy_id = fills[0].strategy_id if fills else "empty"
    return ExecutionResult(
        strategy_id=strategy_id,
        is_bps=is_bps,
        filled_qty=filled_qty,
        unfilled_qty=unfilled_qty,
        fill_count=len(fills),
        avg_markout_500ms_bps=mo,
        vwap_price=vwap,
        arrival_mid=replay.arrival_mid,
        close_mid=replay.close_mid,
    )
