"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model: trophy counts, not rates.  A trophy's rate is E / N, where N is the
players Exophase tracks and E the tracked players holding the trophy.  Between
page refreshes N is fixed and E grows as tracked players earn the trophy
(hazard h per unheld player per hour); at a refresh N jumps and the new
cohort joins holding the trophy at rate a (percent).  Per trophy:

    dE/dt = h * (N - E)          between refreshes
    E    += a/100 * dN           at a refresh

h and a are fitted per trophy from the case history: h from stretches where
N is flat, a from stretches where N jumps.  The refresh process itself
(rate per day, log jump-size mean/sd) is fitted from the jumps in the last
24 h of history.  The forecast is a Monte Carlo over refresh events to the
target (fixed seed, so two runs write identical bytes): the five quantiles
of each trophy's terminal rate over the scenarios.

When the history is flat (no refresh seen, as in the practice cases), the
fitted values are unidentifiable, so priors take over: refreshes about
every 4 h, each multiplying N by ~4x (the launch-night refresh the kit
documents multiplied N by ~6x), a new cohort that holds essentially only
the trophies already at ~100% (the one trophy at 100% stayed at 100%
through that refresh while every other rate fell), and a small earning
hazard.  That prior is what makes the practice forecast expect a refresh
instead of repeating the last rate with a narrow band.

Standard library only, no network, no wall clock, no unseeded randomness.
Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is shipped; kept for parity).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]  # kept for reference; not used below

# Monte Carlo scenarios per case (fixed seed: deterministic output).
N_SCEN = 800
SEED = 20261119
# Cap on events simulated per scenario and on total N growth (guards the
# extrapolation when the fitted refresh process would compound absurdly).
MAX_EVENTS = 60
MAX_N_MULT = 200.0

# Priors, used when the history shows no refresh (flat practice history).
PRIOR_LAMBDA_PER_DAY = 6.0     # about one refresh every 4 h
PRIOR_JUMP_MEDIAN = 6.0         # N multiplier per refresh
PRIOR_JUMP_LOG_SD = 0.6
PRIOR_HAZARD_PER_HOUR = 0.002   # earning hazard when nothing is observed
# Parameter jitter across scenarios (lognormal sd): how unsure the fit is.
JIT_H = 0.4
JIT_A = 0.4


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile_sorted(xs, tau):
    """Linear-interpolated quantile of an already sorted list."""
    n = len(xs)
    if n == 1:
        return xs[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def fit_trophy(history, idx):
    """Fit (h, a) for one trophy from the history reads.

    history: list of dicts with 't' (datetime), 'N' (players), 'E' (holders).
    Returns (h_per_hour, a_percent, last_E, last_N).
    """
    h_cands = []
    a_cands = []
    flat = []
    jumps = []
    for j in range(len(history) - 1):
        r0, r1 = history[j], history[j + 1]
        dt = (r1["t"] - r0["t"]).total_seconds() / 3600.0
        if dt <= 0 or r0["E"] is None or r1["E"] is None:
            continue
        n0, n1 = r0["N"], r1["N"]
        e0, e1 = r0["E"], r1["E"]
        if n0 <= 0 or n1 <= 0:
            continue
        if abs(n1 - n0) <= 0.005 * n0:
            flat.append((dt, n0, e0, e1))
        elif n1 > 1.03 * n0:
            jumps.append((dt, n0, n1, e0, e1))
    for dt, n0, e0, e1 in flat:
        denom = max(n0 - e0, 1.0)
        h_cands.append(min(0.2, max(0.0, (e1 - e0) / (denom * dt))))
    h = median(h_cands)
    for dt, n0, n1, e0, e1 in jumps:
        hh = h if h is not None else 0.0
        prog = hh * max(n0 - e0, 0.0) * dt
        a_cands.append(min(100.0, max(0.0, (e1 - e0 - prog) / max(n1 - n0, 1.0) * 100.0)))
    a = median(a_cands)
    return h, a, flat, jumps


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_h = case["horizon_minutes"] / 60.0
    if horizon_h <= 0:
        raise ValueError(f"{name} has a non-positive horizon")

    # ---- history: times, N, per-trophy E -------------------------------
    hist = []
    for read in case["history"]:
        rates = read["rates"]
        n = read["players_tracked"]
        hist.append({
            "t": parse_time(read["at"]),
            "N": n,
            "E": [None if r is None else r * n / 100.0 for r in rates],
        })
    if not hist:
        raise ValueError(f"{name} has no history")
    last = hist[-1]
    last_n = last["N"]
    span_days = max((hist[-1]["t"] - hist[0]["t"]).total_seconds() / 86400.0, 1e-9)

    # ---- per-trophy (h, a) --------------------------------------------
    per_tr = []
    all_jumps = []  # (t_mid, multiplier) over all trophies' shared N path
    seen_jump_keys = set()
    for j in range(len(hist) - 1):
        n0, n1 = hist[j]["N"], hist[j + 1]["N"]
        if n0 > 0 and n1 > 1.03 * n0:
            key = (j, n0, n1)
            if key not in seen_jump_keys:
                seen_jump_keys.add(key)
                all_jumps.append((hist[j]["t"], n1 / n0))
    hs, aas = [], []
    last_e = []
    for i in range(n_tr):
        sub = [{"t": r["t"], "N": r["N"], "E": r["E"][i]} for r in hist]
        h, a, _, _ = fit_trophy(sub, i)
        if h is not None:
            hs.append(h)
        if a is not None:
            aas.append(a)
        # last observed holders for this trophy (walk back over nulls)
        e_last = None
        for r in reversed(hist):
            if r["E"][i] is not None:
                e_last = r["E"][i]
                break
        last_e.append(0.0 if e_last is None else e_last)
    h_global = median(hs) if hs else None
    if h_global is None:
        h_global = PRIOR_HAZARD_PER_HOUR

    rates_last = []
    for i in range(n_tr):
        r = None
        for read in reversed(case["history"]):
            if read["rates"][i] is not None:
                r = read["rates"][i]
                break
        rates_last.append(r)

    h_list, a_list = [], []
    for i in range(n_tr):
        sub = [{"t": r["t"], "N": r["N"], "E": r["E"][i]} for r in hist]
        h, a, _, _ = fit_trophy(sub, i)
        h_list.append(h if h is not None else h_global)
        if a is not None:
            a_list.append(a)
        else:
            r = rates_last[i]
            a_list.append(100.0 if (r is not None and r >= 99.5) else 0.0)

    # ---- refresh process (lambda, log-jump mean/sd) --------------------
    lam_day, mu, sigma = PRIOR_LAMBDA_PER_DAY, math.log(PRIOR_JUMP_MEDIAN), PRIOR_JUMP_LOG_SD
    t_end = hist[-1]["t"]
    recent = [m for (t, m) in all_jumps
              if (t_end - t).total_seconds() <= 24 * 3600.0]
    pool = recent if len(recent) >= 2 else all_jumps
    if len(pool) >= 2:
        if len(recent) >= 2:
            lam_day = len(recent) / 1.0
        else:
            lam_day = len(all_jumps) / span_days
        lam_day = min(24.0, max(0.5, lam_day))
        logs = [math.log(m) for m in pool]
        mu = sum(logs) / len(logs)
        var = sum((x - mu) ** 2 for x in logs) / max(len(logs) - 1, 1)
        sigma = min(1.0, max(0.3, math.sqrt(var)))
        # Cap the median total N growth over the horizon at 100x: early
        # launch refreshes (a 6x backlog jump) must not compound forever.
        med_growth = lam_day * (horizon_h / 24.0) * mu
        if med_growth > math.log(100.0) and mu > 0:
            mu *= math.log(100.0) / med_growth

    lam_h = lam_day / 24.0

    # ---- Monte Carlo ----------------------------------------------------
    rng = random.Random(SEED)
    terminal = [[] for _ in range(n_tr)]  # terminal rates per trophy
    for _ in range(N_SCEN):
        # one refresh-event path for this scenario
        events = []
        t = 0.0
        n_mult = 1.0
        while len(events) < MAX_EVENTS:
            t += rng.expovariate(lam_h) if lam_h > 0 else float("inf")
            if t >= horizon_h or t == float("inf"):
                break
            m = math.exp(rng.gauss(mu, sigma))
            m = min(8.0, max(1.02, m))
            if n_mult * m > MAX_N_MULT:
                break
            n_mult *= m
            events.append((t, m))
        for i in range(n_tr):
            h = h_list[i] * math.exp(rng.gauss(0.0, JIT_H))
            base_a = a_list[i]
            if base_a >= 99.5:
                # a trophy every new player already holds stays held: the
                # documented refresh left the 100% trophy at 100%.
                a = 100.0
            elif base_a <= 0.0:
                a = 0.0
            else:
                a = min(100.0, max(0.0, base_a * math.exp(rng.gauss(0.0, JIT_A))))
            e = last_e[i]
            n = float(last_n)
            t_prev = 0.0
            for (te, m) in events:
                dt = te - t_prev
                if dt > 0 and h > 0 and n > e:
                    e = n - (n - e) * math.exp(-h * dt)
                dn = n * (m - 1.0)
                e += a / 100.0 * dn
                n *= m
                t_prev = te
            dt = horizon_h - t_prev
            if dt > 0 and h > 0 and n > e:
                e = n - (n - e) * math.exp(-h * dt)
            terminal[i].append(100.0 * e / n if n > 0 else 0.0)

    rows = []
    for i, trophy in enumerate(trophies):
        vals = sorted(terminal[i])
        qs = [round4(clamp(quantile_sorted(vals, tau))) for tau in LEVELS]
        # enforce non-decreasing after rounding
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
