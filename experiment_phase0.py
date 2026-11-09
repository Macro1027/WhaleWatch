"""
Phase 0 experiment.

Question: does a Toxicity-Gated TWAP beat Baseline TWAP by >=2 bps of
side-aware implementation shortfall on a synthetic whale-attack replay,
robustly across >=3 threshold values and without winning via unfilled inventory?

Scenario: BUY 2 BTC over 600s (10 min) in 20 slices of 0.1 BTC.
A scripted whale (aggressor-side burst) hits mid-replay.
Terminal inventory charged at close mid (opportunity cost vs arrival).
1 bp taker fee on filled portion.

Signal: 1s rolling signed trade imbalance, z-scored over trailing 30s.
Strictly causal.

The only random inputs are per-seed GBM mid noise and Poisson trade counts.
No code receives the whale flag or any future value as input.
"""
from __future__ import annotations
import numpy as np

# ------------------ market config ------------------
DURATION_S = 600
DT = 0.1
N_STEPS = int(DURATION_S / DT)
BASE_MID = 60_000.0
VOL_BPS_PER_SEC = 2.0           # GBM vol scale for mid
SPREAD_HALF_BPS = 1.0           # 1 bp half-spread -> 2 bp quoted spread
FEE_BPS = 1.0                   # taker fee per filled unit

# whale schedule (NOT provided to any strategy or signal)
WHALE_START_S = 180
WHALE_END_S = 270                # 90s burst
WHALE_PEAK_IMPACT_BPS = 50.0     # mid elevation at peak
POST_WHALE_REVERT_FRAC = 0.5     # fraction of peak that reverts after burst

# trade intensity
BASE_TRADE_LAMBDA = 2.0          # trades/sec, baseline
WHALE_TRADE_LAMBDA = 20.0        # trades/sec, during whale
WHALE_BUY_FRAC = 0.90            # fraction of whale trades that are buy-aggressor

# execution config
TARGET_QTY = 2.0
SLICE_COUNT = 20
SLICE_QTY = TARGET_QTY / SLICE_COUNT
SLICE_INTERVAL_S = DURATION_S / SLICE_COUNT  # 30s

# signal config
IMB_WIN_STEPS = 10               # 1s = 10 steps at DT=0.1
COOLDOWN_STEPS = 50              # 5s sustained sub-threshold required before firing backlog
# Signal is RAW signed-trade-imbalance over the last 1s. Thresholds are absolute
# counts (not z-scores) so the signal cannot be renormalised away by a long
# regime shift (the failure mode seen with 30s rolling z against a 90s whale).
# Base-rate 1s imbalance ~ 0 +- sqrt(2) (Poisson(2,0.5)). Whale 1s imbalance
# ~ 20 * (0.9 - 0.1) = 16. So any threshold in [3, 15] cleanly separates.

# experiment grid
THRESHOLDS = [3.0, 5.0, 7.0, 10.0, 15.0]
SEEDS = [0, 1, 2, 3, 4]

# pass criteria
WIN_BPS = 2.0                    # gated must beat baseline by this much (IS lower)
WIN_SEED_FRAC = 3                # of 5 seeds
INV_CAP_FRAC = 0.10              # gated unfilled <= 10% of target
PASS_THRESHOLDS_REQUIRED = 3     # must hold on >=3 of 5 thresholds


# ------------------ synthetic market ------------------
def simulate_market(seed: int):
    """Return t, mid, buy_trades_per_step, sell_trades_per_step."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_STEPS) * DT

    # GBM mid
    vol_per_step = (VOL_BPS_PER_SEC / 10_000) * np.sqrt(DT)
    log_increments = rng.normal(0.0, vol_per_step, N_STEPS)
    log_mid = np.cumsum(log_increments)

    # whale mid overlay (deterministic path; adds elevation then partial revert)
    whale_overlay = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        ti = t[i]
        if ti < WHALE_START_S:
            whale_overlay[i] = 0.0
        elif ti < WHALE_END_S:
            ramp = (ti - WHALE_START_S) / max(1e-9, (WHALE_END_S - WHALE_START_S) * 0.25)
            whale_overlay[i] = (WHALE_PEAK_IMPACT_BPS / 10_000) * min(1.0, ramp)
        else:
            whale_overlay[i] = (WHALE_PEAK_IMPACT_BPS / 10_000) * (1 - POST_WHALE_REVERT_FRAC)

    mid = BASE_MID * np.exp(log_mid + whale_overlay)

    # Poisson trade counts per step; whale window = higher intensity, buy-skewed
    buy_trades = np.zeros(N_STEPS)
    sell_trades = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        ti = t[i]
        if WHALE_START_S <= ti < WHALE_END_S:
            lam, p_buy = WHALE_TRADE_LAMBDA, WHALE_BUY_FRAC
        else:
            lam, p_buy = BASE_TRADE_LAMBDA, 0.5
        n = rng.poisson(lam * DT)
        n_buy = rng.binomial(n, p_buy)
        buy_trades[i] = n_buy
        sell_trades[i] = n - n_buy

    return t, mid, buy_trades, sell_trades


# ------------------ signal (strictly causal) ------------------
def toxicity_signal(buy_trades: np.ndarray, sell_trades: np.ndarray) -> np.ndarray:
    """Raw signed trade imbalance over the trailing 1s. Causal only."""
    signed = buy_trades - sell_trades
    cs = np.concatenate([[0.0], np.cumsum(signed)])
    imb = np.zeros_like(signed)
    for i in range(len(signed)):
        lo = max(0, i - IMB_WIN_STEPS + 1)
        imb[i] = cs[i + 1] - cs[lo]
    return imb


# ------------------ execution ------------------
def ask(mid: float) -> float:
    return mid * (1 + SPREAD_HALF_BPS / 10_000)


def baseline_twap(t: np.ndarray, mid: np.ndarray):
    """Fire one 0.1 BTC marketable buy at each of 20 scheduled times."""
    fills = []
    for k in range(SLICE_COUNT):
        target_t = k * SLICE_INTERVAL_S
        idx = min(int(target_t / DT), N_STEPS - 1)
        fills.append((t[idx], SLICE_QTY, ask(mid[idx])))
    return fills, 0.0


def gated_twap(t: np.ndarray, mid: np.ndarray, z: np.ndarray, threshold: float):
    """
    Same 20-slice schedule, but at each step:
      - if a scheduled slice time has arrived, add SLICE_QTY to pending backlog
      - fire the entire backlog only if signal has been <= threshold for the last
        COOLDOWN_STEPS steps (hysteresis prevents firing into transient dips
        inside a sustained toxic episode)
    Deadline = last step; any still-pending qty is terminal inventory.
    """
    fills = []
    pending = 0.0
    next_slice = 0
    steps_below = 0  # consecutive steps with z <= threshold
    for i in range(N_STEPS):
        if z[i] <= threshold:
            steps_below += 1
        else:
            steps_below = 0
        if next_slice < SLICE_COUNT and t[i] >= next_slice * SLICE_INTERVAL_S - 1e-9:
            pending += SLICE_QTY
            next_slice += 1
        if pending > 0 and steps_below >= COOLDOWN_STEPS:
            fills.append((t[i], pending, ask(mid[i])))
            pending = 0.0
    return fills, pending


# ------------------ metrics ------------------
def side_aware_is_bps(
    fills,
    unfilled_qty: float,
    arrival_mid: float,
    close_mid: float,
) -> tuple[float, float]:
    """
    Buy-side IS. Positive = cost.
      cost = sum(fill_px * fill_qty) + close_mid * unfilled_qty - arrival_mid * TARGET_QTY
      IS   = 10_000 * cost / (arrival_mid * TARGET_QTY)
    Plus 1 bp taker fee on filled fraction of target.
    """
    filled_qty = sum(q for _, q, _ in fills)
    filled_cost = sum(p * q for _, q, p in fills)
    unfilled_cost = close_mid * unfilled_qty
    excess = filled_cost + unfilled_cost - arrival_mid * TARGET_QTY
    is_bps = 10_000 * excess / (arrival_mid * TARGET_QTY)
    fee_bps = FEE_BPS * (filled_qty / TARGET_QTY)
    return is_bps + fee_bps, filled_qty


# ------------------ driver ------------------
def main():
    print("=" * 92)
    print(f"Phase-0 experiment  |  BUY {TARGET_QTY} BTC over {DURATION_S}s "
          f"in {SLICE_COUNT} slices of {SLICE_QTY:.2f} BTC")
    print(f"Whale: t={WHALE_START_S}..{WHALE_END_S}s, "
          f"peak +{WHALE_PEAK_IMPACT_BPS:.0f}bp, revert {POST_WHALE_REVERT_FRAC*100:.0f}%")
    print(f"Signal: raw 1s signed trade imbalance (absolute thresholds). "
          f"Fee {FEE_BPS:.0f}bp. Half-spread {SPREAD_HALF_BPS:.0f}bp.")
    print("=" * 92)

    results: dict[tuple[float, int], dict] = {}
    for seed in SEEDS:
        t, mid, bt, st = simulate_market(seed)
        z = toxicity_signal(bt, st)
        arrival_mid = float(mid[0])
        close_mid = float(mid[-1])

        bl_fills, bl_unf = baseline_twap(t, mid)
        bl_is, _ = side_aware_is_bps(bl_fills, bl_unf, arrival_mid, close_mid)

        for thr in THRESHOLDS:
            g_fills, g_unf = gated_twap(t, mid, z, thr)
            g_is, _ = side_aware_is_bps(g_fills, g_unf, arrival_mid, close_mid)
            results[(thr, seed)] = dict(
                g_is=g_is, bl_is=bl_is, diff=g_is - bl_is,
                g_unf=g_unf, g_unf_frac=g_unf / TARGET_QTY,
                close_vs_arrival_bps=10_000 * (close_mid - arrival_mid) / arrival_mid,
            )

    # IS-diff table (gated - baseline, negative = gated wins)
    print(f"\n{'Thr':>5} | " + " | ".join(f"S{s} diff" for s in SEEDS) +
          f" | {'mean':>7} | {'wins>=2bp':>9}")
    print("-" * 92)
    for thr in THRESHOLDS:
        diffs = [results[(thr, s)]['diff'] for s in SEEDS]
        wins = sum(1 for d in diffs if d <= -WIN_BPS)
        row = " | ".join(f"{d:+6.2f}" for d in diffs)
        print(f"{thr:>5.1f} | {row} | {np.mean(diffs):+7.2f} | {wins}/5")

    # Unfilled inventory table
    print(f"\n{'Thr':>5} | " + " | ".join(f"S{s} unf%" for s in SEEDS) + "    (gated unfilled fraction)")
    print("-" * 92)
    for thr in THRESHOLDS:
        row = " | ".join(f"{results[(thr, s)]['g_unf_frac']*100:5.1f}%" for s in SEEDS)
        print(f"{thr:>5.1f} | {row}")

    # Raw IS levels (sanity: are these numbers plausible?)
    print(f"\n{'Thr':>5} | " + " | ".join(f"S{s} bl/gt" for s in SEEDS))
    print("-" * 92)
    for thr in THRESHOLDS:
        parts = []
        for s in SEEDS:
            r = results[(thr, s)]
            parts.append(f"{r['bl_is']:+5.1f}/{r['g_is']:+5.1f}")
        print(f"{thr:>5.1f} | " + " | ".join(f"{p:>12}" for p in parts))

    print(f"\n{'close_mid - arrival_mid (bps), per seed':<60}")
    print("  " + " | ".join(f"S{s}: {results[(THRESHOLDS[0], s)]['close_vs_arrival_bps']:+6.2f}" for s in SEEDS))

    # Verdict
    print("\n" + "=" * 92)
    print(f"Verdict criteria:")
    print(f"  per-threshold: >={WIN_SEED_FRAC}/5 seeds show gated IS <= baseline IS - {WIN_BPS:.1f}bp")
    print(f"                 AND gated unfilled <= {INV_CAP_FRAC*100:.0f}% of target on all 5 seeds")
    print(f"  overall PASS:  >={PASS_THRESHOLDS_REQUIRED}/5 thresholds satisfy the above")
    print("-" * 92)

    passing = 0
    for thr in THRESHOLDS:
        wins = sum(1 for s in SEEDS if results[(thr, s)]['diff'] <= -WIN_BPS)
        inv_ok = all(results[(thr, s)]['g_unf_frac'] <= INV_CAP_FRAC for s in SEEDS)
        tag = "ok" if (wins >= WIN_SEED_FRAC and inv_ok) else "--"
        print(f"  thr={thr:.1f}: wins={wins}/5  inv_ok={inv_ok}  [{tag}]")
        if wins >= WIN_SEED_FRAC and inv_ok:
            passing += 1

    print("-" * 92)
    if passing >= PASS_THRESHOLDS_REQUIRED:
        verdict = "PASS -> proceed with Phase A unchanged"
    elif passing >= 1:
        verdict = "MARGINAL -> revise plan (pivot to risk/completion trade-off framing)"
    else:
        verdict = "FAIL -> use fallback (Pareto-frontier demo story)"
    print(f"  passing thresholds: {passing}/5  =>  {verdict}")
    print("=" * 92)


if __name__ == "__main__":
    main()
