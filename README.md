[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Iyannn11/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Iyannn11/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target instant — five quantiles (5% / 25% /
50% / 75% / 95%) per trophy. Entry for the
[make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Model

A trend model, not a hold-last-rate model. For each trophy, `main.ts` fits a
straight line to `logit(rate)` against `ln(1 + hours since launch
2026-11-19T00:00:00Z)` over the case's reads (least squares) and reads the
line at the target. The spread combines the random-walk spread
(`0.5 * sqrt(days)` on the logit scale) with the fit's residual scatter, so
trophies whose history wobbles get wider bands. With fewer than 3 usable
reads it falls back to the last rate with the random-walk spread. Rates are
clamped to 0.01–99.99% before the logit, null rates in early reads are
skipped, and quantiles are rounded to 4 decimals. Fully deterministic — no
randomness, so two runs write byte-identical output.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` JSON forecast per case (`{"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}]}`). Node 22, standard library only, no network.

## License

MIT — see [LICENSE](LICENSE).
