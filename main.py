# main.py — make.fun gta6-trophy-rarity-forecast-01 entry.
# Forecasts each PS5 trophy's Exophase earn rate at the case target as five
# quantiles (0.05, 0.25, 0.5, 0.75, 0.95), in percent. Python 3.13 stdlib only,
# no network, deterministic, well under the 8 s per-case budget.
#
# Model (per trophy, logit scale):
#   * median: damped trend extrapolation of logit(rate) against ln(1+hours
#     since release), fitted by recency-weighted least squares on the recent
#     window; damping halves the drift every ~2 days so a week-long horizon
#     does not run a short-term slope off the rails.
#   * spread: diffusion estimated from the trophy's own observed logit
#     increments (scaled by sqrt of elapsed days), plus a trend-uncertainty
#     term growing with the horizon, plus a first-refresh allowance while the
#     tracked-player count has never moved (Exophase refreshes in jumps and
#     the launch-night page sat still for hours before the first one).
#   * always-100% trophies stay pinned near the clamp ceiling.
import json, math, os, glob

Z = (-1.6449, -0.6745, 0.0, 0.6745, 1.6449)
RELEASE = "2026-11-19T00:00:00Z"


def parse_iso(s):
    # ISO-8601 UTC, fractional seconds optional; returns epoch seconds.
    s = s.replace("Z", "+00:00")
    import datetime
    return datetime.datetime.fromisoformat(s).timestamp()


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def wls_fit(xs, ys, ws):
    # weighted least squares line y = a + b x; returns (a, b, resid_sigma).
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sigma = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / sw)
    return a, b, sigma


def forecast_case(case):
    release = parse_iso(case.get("game", {}).get("release_date") or RELEASE)
    target = parse_iso(case["target"])
    at = parse_iso(case["at"])
    hist = case.get("history") or []
    trophies = case["trophies"]
    n = len(trophies)
    d_days = max(1e-6, (target - at) / 86400.0)

    reads = []
    for h in hist:
        t = parse_iso(h["at"])
        reads.append((t, h.get("players_tracked") or 0, h.get("rates") or []))
    reads.sort(key=lambda r: r[0])
    players_moved = len({r[1] for r in reads}) > 1

    hours_target = max(0.0, (target - release) / 3600.0)
    g_target = math.log1p(hours_target)

    rows = []
    for i, tro in enumerate(trophies):
        series = [(t, r[i]) for t, _p, r in reads if i < len(r) and r[i] is not None]
        if not series:
            rows.append({"id": tro["id"], "q": [0.01, 0.05, 0.1, 0.5, 2.0]})
            continue
        last_t, last_rate = series[-1]
        if all(r >= 99.99 for _t, r in series):
            rows.append({"id": tro["id"], "q": [99.9, 99.95, 99.99, 99.99, 99.99]})
            continue

        ls = [(t, logit(r)) for t, r in series]
        # recent window: last 36 h of reads (or everything if shorter)
        cutoff = last_t - 36 * 3600
        win = [(t, l) for t, l in ls if t >= cutoff] or ls[-2:]
        g_last = math.log1p(max(0.0, (last_t - release) / 3600.0))
        xs = [math.log1p(max(0.0, (t - release) / 3600.0)) for t, _l in win]
        ys = [l for _t, l in win]
        span = max(1e-9, xs[-1] - xs[0])
        ws = [0.25 + 0.75 * (x - xs[0]) / span for x in xs]
        a, b, fit_sigma = wls_fit(xs, ys, ws)

        # damped extrapolation from the last observed logit
        damping = 1.0 / (1.0 + d_days / 2.0)
        med = ys[-1] + b * (g_target - g_last) * damping
        # tier ceilings keep a short-term slope from promising the impossible
        ceil = {"platinum": 12.0, "gold": 45.0, "silver": 70.0, "bronze": 92.0}.get(tro.get("tier"), 92.0)
        med = min(med, logit(ceil))

        # diffusion: realized volatility over ALL consecutive reads (zeros
        # included — Exophase sits still between refreshes, and diluting the
        # jumps by the quiet reads is what makes the estimate honest)
        sq = []
        for (t0, l0), (t1, l1) in zip(ls, ls[1:]):
            dt = (t1 - t0) / 86400.0
            if dt > 1e-6:
                sq.append((l1 - l0) ** 2 / dt)
        if sq:
            sigma_obs = math.sqrt(sum(sq) / len(sq))
            sigma_obs = min(2.5, max(0.25, sigma_obs))
        else:
            sigma_obs = 0.35
        s = math.sqrt((sigma_obs ** 2) * d_days + (fit_sigma * d_days * 0.5) ** 2)
        if not players_moved:
            s += 1.15  # first-refresh risk: the page can jump a whole logit unit
        s = min(4.0, max(0.15, s))

        q = [round(min(100.0, max(0.0, sigmoid(med + z * s) * 100.0)), 4) for z in Z]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tro["id"], "q": q})
    return {"trophies": rows}


def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(outdir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(indir, "*.in"))):
        case = json.load(open(path, encoding="utf-8"))
        out = forecast_case(case)
        name = os.path.basename(path)[:-3] + ".out"
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
