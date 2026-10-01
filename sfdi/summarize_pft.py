"""
summarize_pft.py -- inspect a pft_latest.json run before trusting it.

    python summarize_pft.py /tmp/pft_test.json

Answers three questions that have to be settled before PFT goes on the map:

  1. What does the distribution look like, against the paper's published
     anchors? Bands drawn without this are invented.
  2. What is DRIVING the extremes? PFT = 0.3 * z_fc^2 * U_ML * dtheta_fc, and
     z_fc is squared, so a low value can come from genuinely conducive air or
     from a z_fc that collapsed for a bad reason. The term breakdown tells the
     two apart.
  3. Are the diagnosed mixed layers actually mixed? ml_theta_range_k is the
     spread of potential temperature across the layer. A real mixed layer holds
     it near constant; a large spread means the average ran through stable air,
     which is what made the one failing validation case run high.

This deliberately does NOT pick display bands. It prints what is needed to
choose them from data.
"""

from __future__ import annotations

import json
import sys

import numpy as np

# Published values from Tory & Kepert (2021), for scale.
ANCHORS = [
    ("Sedgerly Rd (weak plume)", 12),
    ("Chisholm afternoon (firestorm)", 100),
    ("Bald Fire", 140),
    ("Sir Ivan 2017 (intense pyroCb)", 300),
    ("Chisholm morning", 520),
    ("Black Saturday evening", 500),
    ("Black Saturday morning (extreme, NOT conducive)", 1240),
]


def dist(name, vals, unit=""):
    if not vals:
        print(f"  {name:16s} (none)")
        return
    a = np.array(vals, dtype=float)
    print(f"  {name:16s} min {a.min():8.2f}  p10 {np.percentile(a, 10):8.2f}  "
          f"med {np.median(a):8.2f}  p90 {np.percentile(a, 90):8.2f}  "
          f"max {a.max():8.2f}  {unit}")


def main(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    fires = d.get("fires", {})
    ok = {k: v for k, v in fires.items() if v.get("status") == "ok"}
    bad = {k: v for k, v in fires.items() if v.get("status") != "ok"}

    print("=" * 78)
    print(f"{d.get('model','?')} {d.get('cycle_utc','?')} "
          f"f{d.get('forecast_hour','?'):02d}  valid {d.get('valid_utc','?')}")
    print(f"elevation source: {d.get('elevation_source','?')}")
    print(f"{len(ok)} fires with a value, {len(bad)} without")
    print("=" * 78)

    if bad:
        print("\nFires without a value:")
        reasons = {}
        for v in bad.values():
            reasons.setdefault(v.get("status", "?"), []).append(
                v.get("note", "")[:70])
        for st, notes in reasons.items():
            print(f"  {st}: {len(notes)}")
            for n in sorted(set(notes))[:3]:
                print(f"      {n}")

    pft_v = [v["pft_gw"] for v in ok.values()]
    print("\nDistributions")
    dist("PFT", pft_v, "GW")
    dist("z_fc", [v["z_fc_km"] for v in ok.values()], "km")
    dist("dtheta_fc", [v["dtheta_fc_k"] for v in ok.values()], "K")
    dist("U_ML", [v["u_ml_ms"] for v in ok.values()], "m/s")
    dist("ML depth", [v["ml_depth_m"] for v in ok.values()], "m")
    dist("q_ML", [v["q_ml_gkg"] for v in ok.values()], "g/kg")
    if any("ml_theta_range_k" in v for v in ok.values()):
        dist("ML theta spread", [v["ml_theta_range_k"] for v in ok.values()], "K")
    dist("levels used", [v["levels_used"] for v in ok.values()], "")
    if any("ml_levels" in v for v in ok.values()):
        dist("ML levels", [v["ml_levels"] for v in ok.values()], "")
    dist("fire elevation", [v["fire_elev_m"] for v in ok.values()], "m")

    # --- what drives the extremes -----------------------------------------
    rows = sorted(ok.items(), key=lambda kv: kv[1]["pft_gw"])
    print("\nLowest PFT (most conducive) -- check these are low for a real reason")
    print(f"  {'fire':26s}{'PFT':>8}{'z_fc':>7}{'dth':>7}{'U':>7}{'q_ML':>7}"
          f"{'MLdep':>7}{'dTh':>6}{'elev':>7}")
    for k, v in rows[:8]:
        print(f"  {v['name'][:25]:26s}{v['pft_gw']:8.1f}{v['z_fc_km']:7.2f}"
              f"{v['dtheta_fc_k']:7.2f}{v['u_ml_ms']:7.1f}{v['q_ml_gkg']:7.2f}"
              f"{v['ml_depth_m']:7.0f}{v.get('ml_theta_range_k', 0):6.1f}"
              f"{v['fire_elev_m']:7.0f}")

    print("\nHighest PFT (least conducive)")
    for k, v in rows[-5:]:
        print(f"  {v['name'][:25]:26s}{v['pft_gw']:8.1f}{v['z_fc_km']:7.2f}"
              f"{v['dtheta_fc_k']:7.2f}{v['u_ml_ms']:7.1f}{v['q_ml_gkg']:7.2f}"
              f"{v['ml_depth_m']:7.0f}{v.get('ml_theta_range_k', 0):6.1f}"
              f"{v['fire_elev_m']:7.0f}")

    # --- how many fires sit on each side of each published anchor ---------
    print("\nWhere this run sits against the paper's published values")
    a = np.array(pft_v)
    for label, gw in sorted(ANCHORS, key=lambda t: t[1]):
        below = int((a <= gw).sum())
        print(f"  {label:48s} {gw:5d} GW   {below:4d} of {len(a)} fires at or below "
              f"({100.0 * below / len(a):5.1f}%)")

    # --- is the mixed layer actually mixed --------------------------------
    if any("ml_theta_range_k" in v for v in ok.values()):
        tr = np.array([v["ml_theta_range_k"] for v in ok.values()])
        print("\nMixed-layer quality (theta spread across the diagnosed layer)")
        for thr in (1.0, 2.0, 3.0, 5.0, 10.0):
            n = int((tr > thr).sum())
            print(f"  spread > {thr:4.1f} K : {n:4d} of {len(tr)} "
                  f"({100.0 * n / len(tr):5.1f}%)")
        print("  A well-mixed layer holds theta nearly constant. Pick the flag")
        print("  threshold where this distribution actually separates, not by eye.")

        # Does a poorly-mixed layer actually bias PFT high, as theorised?
        hi = a[tr > 3.0]
        lo = a[tr <= 3.0]
        if len(hi) >= 3 and len(lo) >= 3:
            print(f"\n  median PFT, spread <= 3 K : {np.median(lo):8.1f} GW  (n={len(lo)})")
            print(f"  median PFT, spread  > 3 K : {np.median(hi):8.1f} GW  (n={len(hi)})")
            print("  If the second is much larger, the not-mixed case really does")
            print("  bias PFT high and the flag is measuring something real.")

    # --- correlation sanity ------------------------------------------------
    z = np.array([v["z_fc_km"] for v in ok.values()])
    print("\nTerm contributions (PFT = 0.3 * z_fc^2 * U_ML * dtheta_fc)")
    for nm, arr, power in (("z_fc^2", z ** 2, 2), ("U_ML",
                           np.array([v["u_ml_ms"] for v in ok.values()]), 1),
                           ("dtheta_fc",
                            np.array([v["dtheta_fc_k"] for v in ok.values()]), 1)):
        spread = arr.max() / max(arr.min(), 1e-9)
        print(f"  {nm:10s} spans {spread:8.1f}x across these fires")
    print("  The term with the widest span is what the display is really showing.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
