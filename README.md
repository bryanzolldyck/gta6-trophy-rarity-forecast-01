# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bryanzolldyck/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bryanzolldyck/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A GTA VI trophy-rarity forecaster (MIT, see LICENSE): given the house's
recording of Exophase's trophy page up to a checkpoint, it forecasts every
trophy's earn rate at the target as five quantiles, from a count-level model
of how trophy rates actually move — not from repeating the last rate.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row
per trophy.

## Approach

- `main.py` is the whole program: Python 3 standard library only, no
  network, no wall clock, fixed-seed Monte Carlo — a pure function of the
  input file, so two runs write identical bytes.
- Count model, not rate model: a trophy's rate is E / N (holders /
  players tracked). Between page refreshes N is fixed and E grows as
  tracked players earn the trophy (hazard h per unheld player per hour);
  at a refresh N jumps and the new cohort joins holding the trophy at
  rate a percent. Per trophy, h is fitted from stretches where N is flat
  and a from stretches where N jumps.
- Refresh process: the jump rate and log jump-size distribution are fitted
  from the jumps in the last 24 h of history; the forecast is a Monte
  Carlo over refresh events to the target, with per-scenario jitter on h
  and a, and the five quantiles of each trophy's terminal rate.
- Flat-history prior: when no refresh is visible (the practice cases),
  refreshes about every 4 h, each multiplying N by ~6x, a new cohort that
  holds essentially only the trophies already at ~100% — the launch-night
  refresh the kit documents (638 → 3,868 players tracked, every rate but
  the one at 100% falling) — so the forecast expects a refresh instead of
  repeating the last rate with a narrow band.

## Local validation

No local grade exists for this competition (the server grades), so the
practice answers were reconstructed from the refresh the kit documents
(04:30Z read: 3,868 players, 4,276 trophies earned; the 100% trophy
unchanged, the new cohort holding only it), under two progression
assumptions (old cohort unchanged / +10%):

- this forecaster: pinball loss 0.72 / 0.66, vs the starter's persistence
  baseline at 1.69 / 1.59 (kit: about 1.51);
- on a synthetic final-like case (2 days of 30-min reads with refreshes,
  4.5-day horizon, known truth): loss 0.27 vs the starter's 0.44.

---

<sub>Entered on [make.fun](https://make.fun/?ref=3np13rck&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
