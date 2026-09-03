"""Which climate-class assignment validates better against gridMET?

The engine hit 0.95+ correlation and near-zero bias using climate classes I
hand-assigned to six sites. The derived map disagrees at three of them. Rather
than argue about which is more defensible, re-run the validation both ways --
the downloaded data is already on disk, so this costs nothing but a minute.

Whichever assignment reproduces gridMET better is the one to build 39 years
of climatology on.

Usage:  python3 compare_climate_classes.py
"""

import os
import time

import numpy as np

import build_climatology as bc
import nfdrs78 as n
import validate_gridmet as v

cc = np.load(os.path.join(bc.META, "climate_class.npy"))
LATS_FULL = np.linspace(49.42083, 25.04583, 585)
LONS_FULL = np.linspace(-124.7875, -67.0375, 1386)
la = np.argwhere((LATS_FULL >= bc.SOUTH) & (LATS_FULL <= bc.NORTH)).ravel()
lo = np.argwhere((LONS_FULL >= bc.WEST) & (LONS_FULL <= bc.EAST)).ravel()
lats = LATS_FULL[la[0]:la[-1] + 1]
lons = LONS_FULL[lo[0]:lo[-1] + 1]


def derived_class(site):
    i = int(np.argmin(np.abs(lats - site["lat"])))
    j = int(np.argmin(np.abs(lons - site["lon"])))
    return float(cc[i, j])


def run_site(site, climat):
    dates, tmmx = v._read_series(site["name"], "tmmx")
    _, tmmn = v._read_series(site["name"], "tmmn")
    _, rmax = v._read_series(site["name"], "rmax")
    _, rmin = v._read_series(site["name"], "rmin")
    _, vs = v._read_series(site["name"], "vs")
    _, pr = v._read_series(site["name"], "pr")
    _, srad = v._read_series(site["name"], "srad")
    _, erc_ref = v._read_series(site["name"], "erc")
    m = min(len(tmmx), len(erc_ref))

    tmax_f = (tmmx[:m] - 273.15) * 9 / 5 + 32
    tmin_f = (tmmn[:m] - 273.15) * 9 / 5 + 32
    ws = vs[:m] * 0.914 * 2.23694
    ppt = pr[:m] / 25.4
    jd = np.array([int(time.strftime("%j", time.strptime(d[:10], "%Y-%m-%d")))
                   for d in dates[:m]])

    st = n.NFDRSState(1, climate_class=np.array([float(climat)]))
    erc = np.zeros(m)
    for i in range(m):
        sow = n.state_of_weather_from_srad(
            np.array([srad[i]]),
            np.array([v._clear_sky_srad(site["lat"], jd[i])]))
        o = n.step(st, dict(
            tmax_f=tmax_f[i:i+1], tmin_f=tmin_f[i:i+1],
            rhmax=rmax[:m][i:i+1], rhmin=rmin[:m][i:i+1],
            tobs_f=tmax_f[i:i+1], rhobs=rmin[:m][i:i+1],
            sow=sow, ppt_in=ppt[i:i+1], ws_mph=ws[i:i+1],
            lat=np.array([site["lat"]]), jday=int(jd[i])))
        erc[i] = o["erc"][0]

    s = slice(v.SPINUP_DAYS, m)
    a, b = erc[s], erc_ref[:m][s]
    k = np.isfinite(a) & np.isfinite(b)
    return (np.mean(a[k] - b[k]), np.mean(np.abs(a[k] - b[k])),
            np.corrcoef(a[k], b[k])[0, 1])


print(f"{'site':<14}{'hand':>5}{'  bias':>8}{'MAE':>7}{'r':>7}   "
      f"{'derv':>5}{'  bias':>8}{'MAE':>7}{'r':>7}   winner")
print("-" * 78)

hb, db, hm, dm = [], [], [], []
for site in v.SITES:
    d = derived_class(site)
    bh, mh, rh = run_site(site, site["climat"])
    if int(d) == site["climat"]:
        print(f"{site['name']:<14}{site['climat']:>5}{bh:>8.2f}{mh:>7.2f}{rh:>7.3f}"
              f"   {'same':>5}{'':>8}{'':>7}{'':>7}   --")
        hb.append(bh); db.append(bh); hm.append(mh); dm.append(mh)
        continue
    bd, md, rd = run_site(site, d)
    better = "derived" if md < mh else "hand"
    print(f"{site['name']:<14}{site['climat']:>5}{bh:>8.2f}{mh:>7.2f}{rh:>7.3f}"
          f"   {d:>5.0f}{bd:>8.2f}{md:>7.2f}{rd:>7.3f}   {better}")
    hb.append(bh); db.append(bd); hm.append(mh); dm.append(md)

print("-" * 78)
print(f"{'MEAN':<14}{'':>5}{np.mean(hb):>8.2f}{np.mean(hm):>7.2f}{'':>7}"
      f"   {'':>5}{np.mean(db):>8.2f}{np.mean(dm):>7.2f}")
print()
print(f"mean |bias|:  hand {np.mean(np.abs(hb)):.2f}   "
      f"derived {np.mean(np.abs(db)):.2f}")
print(f"mean MAE:     hand {np.mean(hm):.2f}   derived {np.mean(dm):.2f}")
print()
if np.mean(dm) < np.mean(hm):
    print("-> The DERIVED map validates better. Build the climatology on it.")
elif np.mean(dm) > np.mean(hm) * 1.15:
    print("-> The HAND assignments validate clearly better. The precipitation\n"
          "   thresholds need adjusting before the long run.")
else:
    print("-> Too close to call. Climate class is a weak lever here, which is\n"
          "   itself useful: it means the choice does not much affect SFDI.")
