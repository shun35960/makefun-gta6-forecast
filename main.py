"""Entry for the make.fun bounty gta6-twitch-ccu-forecast-01 (Python 3.13, standard library only).

Reads every <id>.in in the input directory and writes <id>.out: for hours 2..24 of the release day, five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of the hour's Twitch viewers. The model (model.py) is an analog ensemble over the
case's comparable launches; if it fails for any reason the case falls back to the house persistence forecast, and
every output is sanitised (finite, 0 < q <= 1e9, non-decreasing, 23 rows) before it is written.

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first so model.py beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))


def round2(x: float) -> float | int:
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour1(case: dict) -> float:
    try:
        v = float(next(k for k in case["known"] if k["hour"] == 1)["viewers"])
        if math.isfinite(v):
            return max(1.0, v)
    except Exception:
        pass
    return 1.0


def persistence(p1: float, h: int) -> list[float]:
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def sanitise(qs: list[float]) -> list[float | int]:
    out, prev = [], 0.01
    for q in qs:
        q = q if math.isfinite(q) else prev
        q = min(1e9, max(0.01, q, prev))
        q = round2(q)
        if q < prev:  # rounding can never lower below prev since prev is itself rounded, but be safe
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_rows(case: dict, params: dict | None = None) -> list[dict]:
    p1 = hour1(case)
    try:
        lr = model.forecast_logratios(case, params)
    except Exception:
        lr = None
    rows = []
    for h in HOURS:
        r = lr.get(h) if lr else None
        if r is None or len(r) != 5 or not all(math.isfinite(x) for x in r):
            qs = persistence(p1, h)
        else:
            qs = [p1 * math.exp(max(-30.0, min(30.0, x))) for x in r]
        rows.append({"hour": h, "q": sanitise(qs)})
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            rows = forecast_rows(case)
        except Exception:
            rows = [{"hour": h, "q": sanitise(persistence(1.0, h))} for h in HOURS]
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps({"hours": rows}, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
