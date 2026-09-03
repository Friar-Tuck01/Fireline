"""Inspect what we are actually parsing out of the gridMET CSVs.

Written after the engine's fuel moisture came out impossibly dry on real data
while behaving correctly on synthetic input. When the math is right and the
output is wrong, look at the inputs.

Checks three things that would each produce plausible-but-wrong results:
  1. UNITS -- is temperature really Kelvin, RH really percent, precip mm?
  2. COLUMN -- is _read_series grabbing the value column or a coordinate?
  3. ALIGNMENT -- do all nine variables cover the same dates in the same
     order? The harness slices by position, so a single missing day in one
     variable silently shifts it against the others from then on.

Usage:  python3 inspect_data.py [site_name]
"""

import sys

import numpy as np

import validate_gridmet as v

site = sys.argv[1] if len(sys.argv) > 1 else "reno_nv"

print(f"\n=== raw CSV header, {site} rmax ===")
print("    (values below are PACKED integers; _read_series unpacks them)")
with open(f"work/{site}_rmax.csv") as f:
    for i, line in enumerate(f):
        print("   ", line.rstrip())
        if i >= 3:
            break

print(f"\n=== parsed series (UNPACKED), {site} ===")
print(f"{'var':>6}{'n':>7}{'min':>10}{'mean':>10}{'max':>10}   {'first date':<12}{'last date':<12}")
series, datesets = {}, {}
for k in v.GRIDMET_VARS:
    try:
        d, x = v._read_series(site, k)
    except FileNotFoundError:
        print(f"{k:>6}   missing")
        continue
    series[k], datesets[k] = x, d
    print(f"{k:>6}{len(x):>7}{np.nanmin(x):>10.2f}{np.nanmean(x):>10.2f}"
          f"{np.nanmax(x):>10.2f}   {d[0][:10]:<12}{d[-1][:10]:<12}")

print("\n=== unit sanity ===")
def verdict(label, ok, detail):
    print(f"  {'OK  ' if ok else 'BAD '} {label:<34}{detail}")

if "tmmx" in series:
    t = series["tmmx"]
    verdict("tmmx looks like Kelvin", 250 < np.nanmean(t) < 320,
            f"mean {np.nanmean(t):.1f} (K expected ~280-300, F would be ~60-70)")
if "rmax" in series and "rmin" in series:
    verdict("rmax > rmin everywhere",
            bool(np.all(series["rmax"][:len(series["rmin"])]
                        >= series["rmin"][:len(series["rmax"])])),
            "max RH must exceed min RH on every day")
    verdict("rmax is percent not fraction", np.nanmax(series["rmax"]) > 2.0,
            f"max {np.nanmax(series['rmax']):.1f}")
    verdict("rmin is percent not fraction", np.nanmax(series["rmin"]) > 2.0,
            f"max {np.nanmax(series['rmin']):.1f}")
if "pr" in series:
    verdict("pr looks like mm", np.nanmax(series["pr"]) > 5.0,
            f"max {np.nanmax(series['pr']):.1f} mm")
if "erc" in series:
    verdict("gridmet erc in 0-120", np.nanmax(series["erc"]) < 120,
            f"range {np.nanmin(series['erc']):.0f}-{np.nanmax(series['erc']):.0f}")

print("\n=== date alignment (the silent killer) ===")
lengths = {k: len(d) for k, d in datesets.items()}
print(f"  lengths: {lengths}")
if len(set(lengths.values())) == 1:
    print("  all series same length")
else:
    print("  >>> LENGTHS DIFFER -- positional slicing is misaligning variables")

ref = datesets.get("tmmx")
if ref:
    for k, d in datesets.items():
        nmin = min(len(d), len(ref))
        mism = [i for i in range(nmin) if d[i] != ref[i]]
        if mism:
            print(f"  >>> {k}: first date mismatch at index {mism[0]} "
                  f"({d[mism[0]]} vs {ref[mism[0]]}), {len(mism)} total")
        else:
            print(f"  {k}: aligned with tmmx over {nmin} days")

print("\n=== a hot summer day, all variables together ===")
if ref:
    tm = series["tmmx"]
    summer = [i for i in range(min(len(tm), 2000))
              if ref[i][5:7] in ("07", "08")]
    for i in summer[:3] + summer[-3:]:
        row = {k: (series[k][i] if i < len(series[k]) else float("nan"))
               for k in series}
        tf = (row.get("tmmx", np.nan) - 273.15) * 9 / 5 + 32
        print(f"  {ref[i][:10]}  Tmax={tf:5.1f}F  RHmin={row.get('rmin',np.nan):5.1f}%"
              f"  RHmax={row.get('rmax',np.nan):5.1f}%  pr={row.get('pr',np.nan):5.2f}mm"
              f"  srad={row.get('srad',np.nan):6.1f}  ERC={row.get('erc',np.nan):5.1f}")
