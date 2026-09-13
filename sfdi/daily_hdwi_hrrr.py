"""
Today's HDWI percentile, from HRRR against the HRRR climatology.

The gridMET HDWI layer is ranked against 39 years but shows observations 2-4
days old. This one is current, at the cost of a 12-year basis. Both are
honest; they answer slightly different questions, and the page labels which
climatology each is using.

    python3 daily_hdwi_hrrr.py run

THE SAMPLING MATCHES THE CLIMATOLOGY EXACTLY, WHICH IS THE WHOLE POINT
    The climatology is the daily maximum over four analyses at 00/06/12/18Z.
    So this takes the maximum over the four MOST RECENT such analyses -- a
    rolling 24 hours rather than a calendar day.

    Rolling, not calendar, on purpose. Early in the day a calendar-day
    maximum would have only one or two samples where the climatology had
    four, and a maximum over fewer draws is systematically smaller. Every
    morning would read artificially calm. Calendar alignment buys nothing
    here; equal sample counts buy correctness.

WHY THIS EXISTS AT ALL
    Feeding HRRR into the gridMET climatology was measured and rejected --
    after correcting the definitional mismatch, only 30% of values landed
    within 5 percentile points, with correlation 0.748. That scatter is not
    a fixable offset. A percentile is only meaningful when today's value and
    the climatology come from the same source, so the climatology had to be
    rebuilt from HRRR.
"""

import datetime as dt
import json
import os
import sys

import numpy as np

import build_climatology as bc
import build_hdwi_hrrr_climatology as hc

OUT_DIR = os.path.join(bc.WORK, "daily")

# Same colours and the same meaning as the gridMET HDWI percentile layer.
# The climatologies differ; the bands do not.
BANDS = [
    (90, (250, 204,  60, 150), '90-95th', 0.05),
    (95, (245, 130,  30, 180), '95-97th', 0.02),
    (97, (215,  45,  35, 210), '97-99th', 0.02),
    (99, (255, 255, 255, 240), '99th+',   0.01),
]


def recent_analyses(n=4):
    """The n most recent 00/06/12/18Z analyses that should be published.

    HRRR appears roughly 50-90 minutes after its cycle, so anything newer
    than ~2 hours is a gamble. Walking back from there gives a rolling
    24-hour window with the same four-sample spacing the climatology used.
    """
    t = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    out = []
    probe = t.replace(minute=0, second=0, microsecond=0)
    while len(out) < n and (t - probe) < dt.timedelta(hours=36):
        if probe.hour in hc.SAMPLE_HOURS:
            out.append(probe)
        probe -= dt.timedelta(hours=1)
    return out


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    lut_path = os.path.join(bc.WORK, "breakpoints", "hdwi_hrrr_lookup.npz")
    for cand in (lut_path, "hdwi_hrrr_lookup.npz",
                 os.path.join("data", "hdwi_hrrr_lookup.npz")):
        if os.path.exists(cand):
            lut = np.load(cand)
            break
    else:
        sys.exit("hdwi_hrrr_lookup.npz not found — "
                 "run build_hdwi_hrrr_climatology.py")
    curves, pcts = lut["curves"], list(lut["pcts"])

    nn = hc._build_nn()
    best = None
    used = []
    for c in recent_analyses():
        try:
            v = hc.hrrr_hdwi(c.date(), nn, hour=c.hour)
        except Exception as e:
            print(f"   {c:%Y-%m-%d %HZ} unavailable ({str(e)[:50]})")
            continue
        best = v if best is None else np.maximum(best, v)
        used.append(c)
        print(f"   {c:%Y-%m-%d %HZ}  mean HDWI {v.mean():6.1f}")
    if best is None:
        sys.exit("no HRRR analyses retrieved")
    if len(used) < 4:
        # Fewer samples means a systematically smaller maximum, which biases
        # the percentile low. Worth saying out loud rather than quietly
        # publishing a calmer-looking map.
        print(f"   NOTE: only {len(used)} of 4 analyses available; today's "
              f"maximum is drawn from fewer samples than the climatology "
              f"and will read slightly low")

    cls = np.zeros(best.size, dtype=np.int8)
    for i, (p, _c, _n, _e) in enumerate(BANDS, start=1):
        thr = curves[:, pcts.index(float(p))]
        cls = np.where((best > thr) & (thr > 0.01), i, cls)

    from PIL import Image
    cc = np.load(os.path.join(bc.META, "climate_class.npy"))
    shape2d = cc.shape
    land = np.isfinite(cc).ravel()
    rgba = np.zeros((best.size, 4), dtype=np.uint8)
    for i, (_p, col, _n, _e) in enumerate(BANDS, start=1):
        rgba[(cls == i) & land] = col
    png = os.path.join(OUT_DIR, "hdwi_hrrr_latest.png")
    Image.fromarray(rgba.reshape(shape2d + (4,)), "RGBA").save(png, optimize=True)

    nland = int(land.sum())
    counts = {n: int(((cls == i+1) & land).sum())
              for i, (_p, _c, n, _e) in enumerate(BANDS)}
    above90 = int(((cls >= 1) & land).sum())
    lats = np.load(os.path.join(bc.META, "lat.npy"))
    lons = np.load(os.path.join(bc.META, "lons.npy"))
    with open(os.path.join(OUT_DIR, "hdwi_hrrr_latest.json"), "w") as f:
        json.dump({
            "valid_date": used[0].date().isoformat(),
            "window_end_utc": used[0].isoformat().replace("+00:00", "Z"),
            "analyses_used": [c.isoformat().replace("+00:00", "Z")
                              for c in used],
            "generated_utc": dt.datetime.now(dt.timezone.utc)
                               .isoformat().replace("+00:00", "Z"),
            "bounds": [[float(lats.min()), float(lons.min())],
                       [float(lats.max()), float(lons.max())]],
            "bands": [n for _p, _c, n, _e in BANDS],
            "colors": ["#%02X%02X%02X" % c[:3] for _p, c, _n, _e in BANDS],
            "band_cell_counts": counts,
            "above_p90_cells": above90,
            "above_p90_fraction": round(above90 / nland, 5),
            "climatology": f"HRRR {int(lut['years'][0])}-{int(lut['years'][1])}, "
                           f"{int(lut['ndays'][0]):,} days",
            "climatology_years": 12,
            "aggregation": "max over the four most recent 00/06/12/18Z analyses",
            "source": "HDWI = wind x VPD from HRRR 3 km, ranked against a "
                      "per-cell HRRR climatology",
        }, f, indent=2)

    print(f"\nHDWI percentile (HRRR) — rolling 24 h ending "
          f"{used[0]:%Y-%m-%d %HZ}")
    for _p, _c, n, expect in BANDS:
        print(f"   {n:>8}: {counts[n]:>8,} cells {100*counts[n]/nland:>5.2f}%  "
              f"(average day {100*expect:>4.0f}%)")
    print(f"   {'above p90':>8}: {above90:>8,} cells "
          f"{100*above90/nland:>5.2f}%  (average day   10%)")


if __name__ == "__main__":
    run()
