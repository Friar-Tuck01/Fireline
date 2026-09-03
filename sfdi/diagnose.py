"""Locate the source of the ERC bias, rather than guessing at it.

Prints the fuel-moisture state alongside the ERC disagreement, broken out by
month. The shape of the bias tells us which subsystem is wrong:

  * bias roughly CONSTANT year-round  -> a scaling/unit problem, or fine-fuel
    moisture running dry every day (insolation correction, 1300 LST proxy)
  * bias LARGE IN SPRING, small in late summer -> live fuel moisture never
    greens up, so the model thinks the herbaceous layer is cured all year
  * bias tracks MC1000 -> the 1000-hour recursion is off
  * our ERC pinned near its ceiling -> saturation, which also explains a
    depressed correlation because variance gets compressed

Usage:  python3 diagnose.py [site_name]
"""

import sys
import time

import numpy as np

import nfdrs78 as n
import validate_gridmet as v

site_name = sys.argv[1] if len(sys.argv) > 1 else "reno_nv"
site = next(s for s in v.SITES if s["name"] == site_name)

dates, tmmx = v._read_series(site_name, "tmmx")
_, tmmn = v._read_series(site_name, "tmmn")
_, rmax = v._read_series(site_name, "rmax")
_, rmin = v._read_series(site_name, "rmin")
_, vs = v._read_series(site_name, "vs")
_, pr = v._read_series(site_name, "pr")
_, srad = v._read_series(site_name, "srad")
_, erc_ref = v._read_series(site_name, "erc")
_, bi_ref = v._read_series(site_name, "bi")

m = min(len(tmmx), len(erc_ref))
tmax_f = (tmmx[:m] - 273.15) * 9 / 5 + 32
tmin_f = (tmmn[:m] - 273.15) * 9 / 5 + 32
ws_mph = vs[:m] * 0.914 * 2.23694
ppt_in = pr[:m] / 25.4
months = np.array([int(d[5:7]) for d in dates[:m]])
jdays = np.array([int(time.strftime("%j", time.strptime(d[:10], "%Y-%m-%d")))
                  for d in dates[:m]])

st = n.NFDRSState(1, climate_class=np.array([float(site["climat"])]))
rec = {k: np.zeros(m) for k in
       ("erc", "bi", "mc1", "mc10", "mc100", "mc1000", "mcherb", "mcwood")}
greenup_flag = np.zeros(m)
sow_rec = np.zeros(m)

for i in range(m):
    sow = n.state_of_weather_from_srad(
        np.array([srad[i]]), np.array([v._clear_sky_srad(site["lat"], jdays[i])]))
    sow_rec[i] = sow[0]
    out = n.step(st, dict(
        tmax_f=tmax_f[i:i+1], tmin_f=tmin_f[i:i+1],
        rhmax=rmax[:m][i:i+1], rhmin=rmin[:m][i:i+1],
        tobs_f=tmax_f[i:i+1], rhobs=rmin[:m][i:i+1],
        sow=sow, ppt_in=ppt_in[i:i+1], ws_mph=ws_mph[i:i+1],
        lat=np.array([site["lat"]]), jday=int(jdays[i])))
    for k in rec:
        rec[k][i] = out[k][0]
    greenup_flag[i] = st.greenup[0]

s = slice(v.SPINUP_DAYS, m)
ours, theirs = rec["erc"][s], erc_ref[:m][s]
mo = months[s]

print(f"\n=== {site_name}  (NFDRS climate class {site['climat']}) ===")
print(f"days compared: {len(ours)}\n")

print("OUR ERC vs GRIDMET ERC, by month")
print(f"{'mo':>3}{'ours':>8}{'grid':>8}{'bias':>8}   "
      f"{'MC1':>6}{'MC10':>6}{'MC100':>7}{'MC1k':>7}{'HERB':>7}{'WOOD':>7}{'green%':>8}")
for mm in range(1, 13):
    k = mo == mm
    if not k.any():
        continue
    kk = np.zeros(m, bool)
    kk[s] = k
    print(f"{mm:>3}{ours[k].mean():>8.1f}{theirs[k].mean():>8.1f}"
          f"{ours[k].mean()-theirs[k].mean():>8.1f}   "
          f"{rec['mc1'][kk].mean():>6.1f}{rec['mc10'][kk].mean():>6.1f}"
          f"{rec['mc100'][kk].mean():>7.1f}{rec['mc1000'][kk].mean():>7.1f}"
          f"{rec['mcherb'][kk].mean():>7.1f}{rec['mcwood'][kk].mean():>7.1f}"
          f"{100*greenup_flag[kk].mean():>7.0f}%")

print("\nDISTRIBUTIONS")
for lbl, arr in (("ours", ours), ("gridmet", theirs)):
    q = np.percentile(arr, [0, 10, 50, 90, 97, 100])
    print(f"  {lbl:>8}  min {q[0]:5.0f}  p10 {q[1]:5.0f}  p50 {q[2]:5.0f}"
          f"  p90 {q[3]:5.0f}  p97 {q[4]:5.0f}  max {q[5]:5.0f}")

ceiling = np.mean(ours >= np.percentile(ours, 99.5) - 1)
print(f"\n  fraction of our days pinned at the ceiling: {100*ceiling:.1f}%")
print(f"  fraction of days greenup was active:       "
      f"{100*greenup_flag[s].mean():.1f}%")
print(f"  state-of-weather mean (0=clear, 3=overcast): {sow_rec[s].mean():.2f}")

# Is the bias flat or seasonal? A flat bias is a scaling problem; a seasonal
# one points at the live fuel models.
bias_by_month = np.array([ours[mo == mm].mean() - theirs[mo == mm].mean()
                          for mm in range(1, 13) if (mo == mm).any()])
print(f"\n  bias range across months: {bias_by_month.min():.1f} to "
      f"{bias_by_month.max():.1f}  (spread {np.ptp(bias_by_month):.1f})")
print("  -> " + ("FLAT: suspect scaling / fine-fuel dryness"
                 if np.ptp(bias_by_month) < 12 else
                 "SEASONAL: suspect live fuel moisture / greenup"))

# Correlation on the seasonal cycle alone vs day-to-day. If the seasonal
# shape is right but daily wiggle is wrong, that is a different bug than
# the reverse.
print(f"\n  r overall:            {np.corrcoef(ours, theirs)[0,1]:.3f}")
clim_o = np.array([ours[mo == mm].mean() for mm in range(1, 13)])
clim_t = np.array([theirs[mo == mm].mean() for mm in range(1, 13)])
print(f"  r of monthly means:   {np.corrcoef(clim_o, clim_t)[0,1]:.3f}")
anom_o = ours - np.array([clim_o[mm-1] for mm in mo])
anom_t = theirs - np.array([clim_t[mm-1] for mm in mo])
print(f"  r of daily anomalies: {np.corrcoef(anom_o, anom_t)[0,1]:.3f}")

print("\nBI check (independent of ERC's loading weighting)")
bours, bthe = rec["bi"][s], bi_ref[:m][s]
print(f"  ours mean {bours.mean():.1f}   gridmet mean {bthe.mean():.1f}"
      f"   r {np.corrcoef(bours, bthe)[0,1]:.3f}")
