# GTA VI Trophy Rarity Forecast — Round 2

Forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles
(5 / 25 / 50 / 75 / 95%) of its Exophase earn rate (%) at
2026-11-26T00:00:00Z, from the house's reads of the trophy page before the
checkpoint.

## Model (per trophy, all on the logit scale)

1. **Recency-weighted trend.** Weighted least-squares line of
   `logit(rate)` against `ln(1 + hours since 2026-11-19T00:00:00Z)`, over
   every non-null read. Recent reads weigh more (linear ramp 0.5 → 1.0
   across the history): the earn-rate curve steepens as launch approaches,
   so launch-night reads should not anchor the trend.
2. **Adaptive trend weight.** The trend centre is blended with the last
   read's rate, with the blend decided by the fit's strength:
   `w = 0.5 + 0.45 · R²` (R² = 0 → 50/50, R² = 1 → 95% trend). A strong,
   clean trend is trusted; a noisy one is shrunk hard toward the last
   observed rate.
3. **Late-surge slope.** When the last 48 h hold ≥ 5 reads spanning ≥ 24 h,
   the slope over that window is blended in (75% full-history slope,
   25% late-window slope), so an acceleration in the final days is partly
   captured without letting a short window dominate.
4. **Per-trophy spread.** The random-walk term `0.5 · sqrt(days to target)`
   combined in quadrature with the fit residual, floored at 0.20 and capped
   at 2.00 logit units. Flat histories keep a sane band; noisy trophies
   widen instead of exploding.
5. **Robustness.** Fewer than 3 usable reads → fall back to the last rate
   with the random-walk spread. Null rates are skipped. Rates are clamped
   to 0.01–99.99 before the logit; forecasts are clamped to 0.01–99.99 and
   rounded to 4 decimals. Fully deterministic, offline, no clock, no
   randomness, no network.

## What changed vs Round 1

Round 1 used a fixed 85% trend / 15% last-rate blend, unweighted
least squares, and a fixed random-walk spread combined with the residual.
Round 2 adds the recency weighting, the R²-adaptive blend, the late-surge
slope blend, and the floored/capped per-trophy spread.

A horizon-scaled residual spread and an inverse-variance trend weight were
both built and tested on synthetic cases with known truth, and both were
rejected: the scaled spread widened the band too far on noisy histories,
and the inverse-variance weight lost overall to the R² weight.

## Verification

- Practice cases `p00` / `p01`: exit 0, 55 trophies each, valid output
  format, byte-identical on re-run, ~0.3 s per run (limit 8 s). On the flat
  practice histories the medians equal the last rates (slope 0, correct).
- Synthetic tests with known truth (rising 1%→2%→4%→8% style, flat, noisy,
  late surge, falling, saturating; final-like horizon of 4.5 days):
  mean pinball loss **Round 2: 0.2404 vs Round 1: 0.2982 — 19.4% better**.
  Biggest win on late-surge histories (0.2690 vs 0.7364); small losses on
  saturating histories (0.3221 vs 0.2944) and noisy histories
  (0.2377 vs 0.2330), where the trend overshoots a flattening curve.

## Run

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` file in the input directory (in name order) and writes
one `<id>.out` per case. Node 22, TypeScript run as-is, standard library
only.
