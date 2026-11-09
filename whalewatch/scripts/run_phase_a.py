"""
Phase-A vertical slice: synthetic replay -> signal -> both strategies -> IS/markout.

Runs the exact scenario validated in experiment_phase0.py, but through the
Phase-B-ready module layout (MarketReplay data contract, Fill/ExecutionResult
dataclasses). Serves as Checkpoint A: the two IS numbers must differ as
expected (gated IS <= baseline IS - 2 bps) on seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.metrics import evaluate
from src.signal.toxicity import trade_imbalance_1s
from src.sim.book import SyntheticMarketConfig, make_synthetic_replay
from src.sim.execution import (
    ExecutionConfig,
    GatedConfig,
    baseline_twap,
    gated_twap,
)


def main() -> int:
    market_cfg = SyntheticMarketConfig()
    exec_cfg = ExecutionConfig(
        target_qty=2.0, slice_count=20, deadline_s=market_cfg.duration_s,
        side="buy", half_spread_bps=1.0,
    )
    gated_cfg = GatedConfig(threshold=7.0, cooldown_steps=50)
    imbalance_window_steps = 10
    fee_bps = 1.0
    seed = 42

    replay = make_synthetic_replay(market_cfg, seed=seed)
    signal = trade_imbalance_1s(
        replay.buy_trades, replay.sell_trades, imbalance_window_steps,
    )

    bl_fills, bl_unf = baseline_twap(replay, exec_cfg)
    g_fills, g_unf = gated_twap(replay, signal, exec_cfg, gated_cfg)

    bl = evaluate(bl_fills, bl_unf, replay, exec_cfg.target_qty, fee_bps, exec_cfg.side)
    gt = evaluate(g_fills, g_unf, replay, exec_cfg.target_qty, fee_bps, exec_cfg.side)

    print(f"Replay source : {replay.source}  seed={seed}  steps={replay.n_steps}")
    print(f"Arrival mid   : {replay.arrival_mid:,.2f}")
    print(f"Close mid     : {replay.close_mid:,.2f}  "
          f"({10_000 * (replay.close_mid - replay.arrival_mid) / replay.arrival_mid:+.2f} bps)")
    print()
    print(f"{'Strategy':<14} {'IS bps':>8} {'Fills':>6} {'VWAP':>12} "
          f"{'Unfilled':>9} {'MO500ms bps':>12}")
    for r in (bl, gt):
        print(f"{r.strategy_id:<14} {r.is_bps:>+8.2f} {r.fill_count:>6} "
              f"{r.vwap_price:>12,.2f} {r.unfilled_qty:>9.3f} "
              f"{r.avg_markout_500ms_bps:>+12.2f}")
    print()
    diff = gt.is_bps - bl.is_bps
    print(f"IS diff (gated - baseline): {diff:+.2f} bps "
          f"(negative = gated saves money)")

    # Checkpoint A acceptance: gated at least 2bp better, no unfilled inventory.
    ok_savings = diff <= -2.0
    ok_inventory = gt.unfilled_qty / exec_cfg.target_qty <= 0.10
    print()
    print(f"Checkpoint A: savings >= 2bp ? {'PASS' if ok_savings else 'FAIL'}   "
          f"unfilled <= 10% ?   {'PASS' if ok_inventory else 'FAIL'}")
    return 0 if (ok_savings and ok_inventory) else 1


if __name__ == "__main__":
    raise SystemExit(main())
