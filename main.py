#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun bounty).

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Standard library only, deterministic: two runs on the same history write the
same forecast byte for byte.

Model
-----
Each game's rank is a drift + diffusion process in rank space:
  * drift: recent rank velocity (recency-weighted least squares over the
    game's rank history) and recent log-playing growth, both shrunk toward
    zero and saturating with horizon (a climb does not run forever);
  * diffusion: Laplace-like tails via a two-component normal mixture, with
    sigma growing like sqrt(hours), so later hours are more diffuse;
  * age-out: a game published 28 days ago leaves the chart; when the
    history carries details.created we stop giving it mass after that hour.
The forecast for a place is the average over the 72 scored hours of the
game's place distribution at that hour, which is exactly what the log
score averages over.  Probability mass a place leaves unnamed is shared
equally over the chart's games not named there, so only games with real
mass are named.
"""
import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned constants (see tuning notes in README) ---
# Each may be overridden by an RFC_* environment variable for local tuning;
# the sandbox sets none of these, so the defaults below are what is graded.
def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default

RHO_V = _envf("RFC_RHO_V", 0.30)       # shrinkage on recent rank velocity (ranks/hour)
RHO_G = _envf("RFC_RHO_G", 8.0)        # ranks per unit log-playing growth per hour
LEVEL_K = _envf("RFC_LEVEL_K", 0.01)   # pull per hour toward the playing-implied rank
AGE_K = _envf("RFC_AGE_K", 0.01)      # downward drift per day a game has been on the chart
TAU = _envf("RFC_TAU", 72.0)           # velocity saturation time (hours)
SIGMA0 = _envf("RFC_SIGMA0", 0.8)      # rank diffusion at horizon 0
SIGMA1 = _envf("RFC_SIGMA1", 0.6)     # rank diffusion growth per sqrt(hour)
TAIL_W = _envf("RFC_TAIL_W", 0.0)     # weight of the wide mixture component
TAIL_F = _envf("RFC_TAIL_F", 3.0)      # wide component sigma multiplier
MASS_CAP = _envf("RFC_MASS_CAP", 0.90)  # max named probability mass per place
MIN_P = _envf("RFC_MIN_P", 0.002)      # only name a game at a place above this
VEL_HALF_LIFE = _envf("RFC_VEL_HALF_LIFE", 48.0)  # recency weight half-life (hours)
MAX_OBS = int(_envf("RFC_MAX_OBS", 12))  # snapshots used per game for fits
TAIL_F = _envf("RFC_TAIL_F", 5.0)      # wide component sigma multiplier
MIN_P = _envf("RFC_MIN_P", 0.005)      # only name a game at a place above this


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bin_prob(mu, sigma, lo, hi):
    return norm_cdf((hi - mu) / sigma) - norm_cdf((lo - mu) / sigma)


def place_prob(mu, sigma, place):
    """P(rank in [place-0.5, place+0.5]) under the two-component mixture.

    Rank 1 is a reflecting barrier: latent strength above the top folds
    back onto place 1 instead of vanishing below rank 0.5.
    """
    lo = place - 0.5
    hi = place + 0.5
    if place == 1:
        lo = -0.5
    p = (1.0 - TAIL_W) * _bin_prob(mu, sigma, lo, hi)
    s2 = sigma * TAIL_F
    p += TAIL_W * _bin_prob(mu, s2, lo, hi)
    return p


def wls_slope(times_h, values):
    """Recency-weighted least-squares slope of values vs time (hours)."""
    if len(times_h) < 2:
        return 0.0
    t_last = times_h[-1]
    ws = [0.5 ** ((t_last - t) / VEL_HALF_LIFE) for t in times_h]
    sw = sum(ws)
    mt = sum(w * t for w, t in zip(ws, times_h)) / sw
    mv = sum(w * v for w, v in zip(ws, values)) / sw
    num = sum(w * (t - mt) * (v - mv) for w, t, v in zip(ws, times_h, values))
    den = sum(w * (t - mt) ** 2 for w, t in zip(ws, times_h))
    if den <= 0:
        return 0.0
    return num / den


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
    window_from = parse_ts(task["window"]["from"])
    n_slots = int(task["window"]["slots"])

    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda r: r["at"])
    if not snaps:
        raise SystemExit("empty history")

    # Per-game observations: (hours since epoch, rank, log playing, created or None)
    obs = {}
    names = {}
    for s in snaps:
        t = parse_ts(s["at"])
        th = t.timestamp() / 3600.0
        details = s.get("details") or {}
        for g in s.get("up_and_coming") or []:
            uid = g["universe_id"]
            names[uid] = g.get("name")
            playing = g.get("playing") or 0
            created = None
            d = details.get(str(uid)) or details.get(uid)
            if isinstance(d, dict) and d.get("created"):
                try:
                    created = parse_ts(d["created"])
                except Exception:
                    created = None
            obs.setdefault(uid, []).append((th, g["rank"], math.log(playing) if playing > 0 else None, created, t))

    last = snaps[-1]
    t_last = parse_ts(last["at"])
    # Cross-sectional fit: rank ~ a + b*log(playing) on the last snapshot.
    pts = [(math.log(g["playing"]), g["rank"]) for g in (last.get("up_and_coming") or [])
           if g.get("playing") and g["playing"] > 0]
    if len(pts) >= 10:
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        den = sum((p[0] - mx) ** 2 for p in pts)
        b_fit = (sum((p[0] - mx) * (p[1] - my) for p in pts) / den) if den > 0 else 0.0
        a_fit = my - b_fit * mx
    else:
        a_fit, b_fit = 0.0, 0.0
    gap_h = (window_from - t_last).total_seconds() / 3600.0
    if gap_h < 0:
        gap_h = 0.0
    chart_last = sorted(last.get("up_and_coming") or [], key=lambda g: g["rank"])

    # Per-game drift / diffusion parameters from the last snapshot's games.
    first_seen = {}
    for s in snaps:
        tt = parse_ts(s["at"])
        for g in s.get("up_and_coming") or []:
            first_seen.setdefault(g["universe_id"], tt)
    games = []
    for g in chart_last:
        uid = g["universe_id"]
        trail = obs.get(uid, [])[-MAX_OBS:]
        times = [x[0] for x in trail]
        ranks = [x[1] for x in trail]
        vel = wls_slope(times, ranks)
        lp = [(x[0], x[2]) for x in trail if x[2] is not None]
        growth = wls_slope([x[0] for x in lp], [x[1] for x in lp]) if len(lp) >= 2 else 0.0
        playing_last = g.get("playing") or 0
        if playing_last > 0 and b_fit != 0.0:
            resid = g["rank"] - (a_fit + b_fit * math.log(playing_last))
        else:
            resid = 0.0
        age_days = (t_last - first_seen.get(uid, t_last)).total_seconds() / 86400.0
        drift = RHO_V * vel - RHO_G * growth - LEVEL_K * resid + AGE_K * age_days
        # Clamp drift: no game climbs/falls faster than 2 ranks/hour sustained.
        drift = max(-2.0, min(2.0, drift))
        created = None
        for x in reversed(trail):
            if x[3] is not None:
                created = x[3]
                break
        exit_h = None
        if created is not None:
            exit_h = (created.timestamp() + 28 * 86400 - t_last.timestamp()) / 3600.0
        games.append({
            "uid": uid,
            "r0": g["rank"],
            "drift": drift,
            "exit_h": exit_h,
        })

    # Average the place distribution over the scored hours.
    probs = {uid: [0.0] * places for uid in [x["uid"] for x in games]}
    mu_avg = {}
    for game in games:
        uid = game["uid"]
        r0 = game["r0"]
        drift = game["drift"]
        exit_h = game["exit_h"]
        mu_sum = 0.0
        for h in range(1, n_slots + 1):
            ht = gap_h + h
            if exit_h is not None and ht > exit_h:
                continue
            h_eff = TAU * (1.0 - math.exp(-ht / TAU))
            mu = r0 + drift * h_eff
            if mu < 1.0:
                mu = 1.0
            sigma = SIGMA0 + SIGMA1 * math.sqrt(ht)
            mu_sum += mu
            for k in range(1, places + 1):
                probs[uid][k - 1] += place_prob(mu, sigma, k)
        n_eff = n_slots
        for k in range(places):
            probs[uid][k] /= n_eff
        mu_avg[uid] = mu_sum / n_eff

    # Name rows above MIN_P; cap each place's named mass at MASS_CAP.
    rows = []
    for k in range(1, places + 1):
        cand = [(probs[uid][k - 1], uid) for uid in probs if probs[uid][k - 1] >= MIN_P]
        total = sum(p for p, _ in cand)
        if total > MASS_CAP and total > 0:
            scale = MASS_CAP / total
            cand = [(p * scale, uid) for p, uid in cand]
        for p, uid in cand:
            rows.append({"universe_id": uid, "place": k, "p": round(p, 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: smallest average predicted rank among games with real top-10 mass.
    ranked = sorted(
        games,
        key=lambda x: (mu_avg[x["uid"]], x["r0"], x["uid"]),
    )
    top10 = [x["uid"] for x in ranked[:places]]

    out = {"probabilities": rows, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps(out, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
