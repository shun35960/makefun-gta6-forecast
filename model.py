"""Analog-ensemble forecaster for gta6-twitch-ccu-forecast-01 (standard library only).

For every past launch (donor) in the case's comparables with a full-enough release day, take its log-ratio curve
r_h = ln(v_h / v_1) for hours h = 2..24 of its UTC release day. Weight donors by similarity to the target
(hour-1 size and the hour-1 vs pre-release momentum features), then form each hour's predictive distribution as a
weighted Gaussian-kernel mixture of the donors' r_h, shrunk toward persistence, with a calibrated spread. The
quantiles are q_tau(h) = p1 * exp(Q_tau(h)).
"""

import json
import math
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Knobs tuned on leave-one-donor-out CV (see the work directory's cv.py). Defaults are the CV winner.
DEFAULT = {
    "bw_size": 1.0,       # kernel width (log units) on hour-1 size
    "bw_mom": 0.6,        # kernel width on ln(v1 / v_prev_hour)
    "bw_lvl": 2.0,        # kernel width on ln(v1 / max of the previous 24 h)
    "w_floor": 0.0,       # weight floor relative to the max weight
    "shrink": 0.9,        # multiply donor log-ratios (1 = no shrink toward persistence)
    "bw0": 0.10,          # kernel bandwidth of each donor's point (log units) at hour 2
    "bw_slope": 0.015,    # extra bandwidth per hour
    "spread": 0.8,        # stretch of quantiles about the median
    "min_points": 20,     # donor needs this many release-day hours (of 24)
    "cls_pen": 1.0,       # weight multiplier for a donor whose unlock type differs (CV: no gain, kept off)
    "spread_slope": 0.0,  # extra quantile stretch per hour beyond hour 2 (CV: no gain, kept off)
}


def json_dumps(x) -> str:
    return json.dumps(x)


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()


def _hour_map(rows, day_start: float, key: str = "from") -> dict:
    """hour offset (0 = first hour of day_start) -> viewers (floored at 1)."""
    out = {}
    for r in rows:
        v = r.get("viewers")
        if v is None:
            continue
        h = int(math.floor((_ts(r[key]) - day_start) / 3600.0 + 1e-9))
        out[h] = max(1.0, float(v))
    return out


def _features(hmap: dict, v1: float):
    """(log size, momentum vs previous hour, level vs previous-24h max). Missing pre-hours fall back sensibly."""
    prev = None
    for h in (-1, -2, -3):
        if h in hmap:
            prev = hmap[h]
            break
    window = [hmap[h] for h in range(-24, 0) if h in hmap]
    mom = math.log(v1 / prev) if prev else 0.0
    lvl = math.log(v1 / max(window)) if window else 0.0
    return math.log(v1), mom, lvl


def unlock_class(text: str) -> int:
    """1 = single simultaneous global unlock, 0 = rolling midnight-local (or unknown)."""
    t = (text or "").lower()
    if "rolling" in t or "midnight local" in t:
        return 0
    return 1 if ("simultaneous" in t or "single global" in t or "single time" in t) else 0


def donors_from_case(case: dict, min_points: int):
    """Each usable comparable: (features, {h: r_h})."""
    out = []
    for cm in case.get("comparables") or []:
        try:
            d0 = _ts(cm["day_start"])
            hm = _hour_map(cm.get("hourly") or [], d0)
        except Exception:
            continue
        if 0 not in hm:
            continue
        day = [h for h in range(1, 24) if h in hm]
        if len(day) + 1 < min_points:
            continue
        v1 = hm[0]
        r = {h + 1: math.log(hm[h] / v1) for h in day}  # contest hour = offset + 1
        out.append((_features(hm, v1), r, unlock_class(cm.get("unlock", ""))))
    return out


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _mixture_quantile(pts, ws, bw, tau):
    lo = min(pts) - 8 * bw - 1.0
    hi = max(pts) + 8 * bw + 1.0
    tw = sum(ws)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        c = sum(w * _norm_cdf((mid - p) / bw) for p, w in zip(pts, ws)) / tw
        if c < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_logratios(case: dict, p: dict | None = None):
    """{hour: [5 log-ratio quantiles]} for the case's forecast hours, or None if no donors."""
    k = dict(DEFAULT)
    if p:
        k.update(p)
    d0 = _ts(case["day_start"])
    v1 = max(1.0, float(next(r for r in case["known"] if r["hour"] == 1)["viewers"]))
    hm = _hour_map(case.get("history_hourly") or [], d0)
    hm[0] = v1
    tf = _features(hm, v1)
    donors = donors_from_case(case, k["min_points"])
    if not donors:
        return None
    tcls = unlock_class(case.get("notes", "") if isinstance(case.get("notes"), str) else json_dumps(case.get("notes")))
    ws = []
    for f, _, dcls in donors:
        z = ((f[0] - tf[0]) / k["bw_size"]) ** 2 + ((f[1] - tf[1]) / k["bw_mom"]) ** 2 + ((f[2] - tf[2]) / k["bw_lvl"]) ** 2
        ws.append(math.exp(-0.5 * z) * (k["cls_pen"] if dcls != tcls else 1.0))
    m = max(ws)
    ws = [max(w / m, k["w_floor"]) for w in ws]
    out = {}
    for row in case["forecast"]:
        h = row["hour"]
        pts, pw = [], []
        for (f, r, _), w in zip(donors, ws):
            if h in r:
                pts.append(k["shrink"] * r[h])
                pw.append(w)
        if not pts or sum(pw) <= 0:
            out[h] = None
            continue
        bw = k["bw0"] + k["bw_slope"] * (h - 2)
        qs = [_mixture_quantile(pts, pw, bw, t) for t in LEVELS]
        med = qs[2]
        sp = k["spread"] * (1.0 + k["spread_slope"] * (h - 2))
        qs = [med + sp * (q - med) for q in qs]
        out[h] = qs
    return out
