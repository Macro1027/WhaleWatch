"""
Market replay container + synthetic generator with a scripted whale injection.

The `MarketReplay` is the shared contract between the synthetic path (this file)
and the real-data path (Phase B's `ingest/replay.py`): parallel timestamped
arrays of mid price and per-step buy/sell aggressor trade counts, plus a
boolean `is_synthetic_event` mask that flags whale-injected steps.

The `is_synthetic_event` mask is for UI display and tests only — it MUST NOT
be consumed by `signal/toxicity.py` or any strategy, or the model-leakage
integrity rule is violated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketReplay:
    """Deterministic event stream for one episode. Immutable snapshot."""
    t_s: np.ndarray                   # shape (N,), seconds since start
    mid: np.ndarray                   # shape (N,), mid price per step
    buy_trades: np.ndarray            # shape (N,), buy-aggressor trade count
    sell_trades: np.ndarray           # shape (N,), sell-aggressor trade count
    is_synthetic_event: np.ndarray    # shape (N,), bool mask
    dt_s: float
    source: str                       # e.g. "synthetic", "binance_ws_replay"

    @property
    def arrival_mid(self) -> float:
        return float(self.mid[0])

    @property
    def close_mid(self) -> float:
        return float(self.mid[-1])

    @property
    def n_steps(self) -> int:
        return int(self.mid.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1]) + self.dt_s


@dataclass(frozen=True)
class SyntheticMarketConfig:
    duration_s: float = 600.0
    dt_s: float = 0.1
    base_mid: float = 60_000.0
    vol_bps_per_sec: float = 2.0
    base_trade_lambda: float = 2.0
    # whale schedule
    whale_start_s: float = 180.0
    whale_end_s: float = 270.0
    whale_peak_impact_bps: float = 50.0
    post_whale_revert_frac: float = 0.5
    whale_trade_lambda: float = 20.0
    whale_buy_frac: float = 0.90
    # ramp: fraction of whale duration over which impact ramps to peak
    whale_ramp_frac: float = 0.25


def make_synthetic_replay(cfg: SyntheticMarketConfig, seed: int) -> MarketReplay:
    """
    Generate a fully deterministic (seed + config) synthetic replay with a
    scripted whale burst. `is_synthetic_event` flags every step that falls
    inside the whale window.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(cfg.duration_s / cfg.dt_s))
    t_s = np.arange(n_steps) * cfg.dt_s

    # GBM mid
    vol_per_step = (cfg.vol_bps_per_sec / 10_000) * np.sqrt(cfg.dt_s)
    log_increments = rng.normal(0.0, vol_per_step, n_steps)
    log_mid = np.cumsum(log_increments)

    # Whale mid overlay: linear ramp to peak, hold, then step-down to residual
    whale_overlay = np.zeros(n_steps)
    whale_mask = np.zeros(n_steps, dtype=bool)
    ramp_duration = (cfg.whale_end_s - cfg.whale_start_s) * cfg.whale_ramp_frac
    for i in range(n_steps):
        ti = float(t_s[i])
        if ti < cfg.whale_start_s:
            whale_overlay[i] = 0.0
        elif ti < cfg.whale_end_s:
            ramp = (ti - cfg.whale_start_s) / max(1e-9, ramp_duration)
            whale_overlay[i] = (cfg.whale_peak_impact_bps / 10_000) * min(1.0, ramp)
            whale_mask[i] = True
        else:
            whale_overlay[i] = (cfg.whale_peak_impact_bps / 10_000) * (1 - cfg.post_whale_revert_frac)

    mid = cfg.base_mid * np.exp(log_mid + whale_overlay)

    # Poisson trade counts per step
    buy_trades = np.zeros(n_steps)
    sell_trades = np.zeros(n_steps)
    for i in range(n_steps):
        if whale_mask[i]:
            lam, p_buy = cfg.whale_trade_lambda, cfg.whale_buy_frac
        else:
            lam, p_buy = cfg.base_trade_lambda, 0.5
        n = int(rng.poisson(lam * cfg.dt_s))
        n_buy = int(rng.binomial(n, p_buy))
        buy_trades[i] = n_buy
        sell_trades[i] = n - n_buy

    return MarketReplay(
        t_s=t_s,
        mid=mid,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        is_synthetic_event=whale_mask,
        dt_s=cfg.dt_s,
        source="synthetic",
    )
