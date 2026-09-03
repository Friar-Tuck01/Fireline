"""Isolate why BI disagrees when ERC agrees.

BI = 3.01 * (SC * ERC)^0.46, so BI has exactly two inputs. Our ERC now
correlates with gridMET at 0.976 while BI sits near 0.67, which means the
fault is almost entirely in SC -- the wind-driven half.

We never see gridMET's SC directly, but we can back it out of the identity:

    SC_implied = (BI / 3.01)^(1/0.46) / ERC

That turns an opaque BI disagreement into a direct SC comparison, and lets
us test the leading suspect: gridMET's `vs` is a DAILY MEAN wind at 10 m,
while NFDRS wants the wind at the 1300 LST observation. A daily mean is both
calmer and smoother than an afternoon peak, which would blunt exactly the
day-to-day variance SC depends on.

Usage:  python3 diagnose_bi.py [site_name]
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
ppt_in = pr[:m] / 25.4
jdays = np.array([int(time.strftime("%j", time.strptime(d[:10], "%Y-%m-%d")))
                  for d in dates[:m]])


def run(wind_mph):
    st = n.NFDRSState(1, climate_class=np.array([float(site["climat"])]))
    sc = np.zeros(m); erc = np.zeros(m); bi = np.zeros(m)
    for i in range(m):
        sow = n.state_of_weather_from_srad(
            np.array([srad[i]]),
            np.array([v._clear_sky_srad(site["lat"], jdays[i])]))
        o = n.step(st, dict(
            tmax_f=tmax_f[i:i+1], tmin_f=tmin_f[i:i+1],
            rhmax=rmax[:m][i:i+1], rhmin=rmin[:m][i:i+1],
            tobs_f=tmax_f[i:i+1], rhobs=rmin[:m][i:i+1],
            sow=sow, ppt_in=ppt_in[i:i+1],
            ws_mph=np.array([wind_mph[i]]),
            lat=np.array([site["lat"]]), jday=int(jdays[i])))
        sc[i] = o["sc"][0]; erc[i] = o["erc"][0]; bi[i] = o["bi"][0]
    return sc, erc, bi


s = slice(v.SPINUP_DAYS, m)
ercg = erc_ref[:m][s]
big = bi_ref[:m][s]

# Back-solve gridMET's implied SC. Only valid where both are positive.
ok = (ercg > 0) & (big > 0)
sc_implied = np.zeros_like(big)
sc_implied[ok] = (big[ok] / 3.01) ** (1 / 0.46) / ercg[ok]

print(f"\n=== {site_name}: where does BI go wrong? ===\n")

# Baseline: our current wind handling.
ws_base = vs[:m] * 0.914 * 2.23694
sc_o, erc_o, bi_o = run(ws_base)
sc_o_s, bi_o_s = sc_o[s], bi_o[s]

print("gridMET's implied SC (back-solved from their BI and ERC)")
q = np.percentile(sc_implied[ok], [10, 50, 90])
print(f"  theirs: p10 {q[0]:.1f}  p50 {q[1]:.1f}  p90 {q[2]:.1f}")
q = np.percentile(sc_o_s[ok], [10, 50, 90])
print(f"  ours:   p10 {q[0]:.1f}  p50 {q[1]:.1f}  p90 {q[2]:.1f}")
print(f"  r(SC ours, SC theirs) = {np.corrcoef(sc_o_s[ok], sc_implied[ok])[0,1]:.3f}")

print("\nIs their SC even responding to gridMET's own wind?")
w = ws_base[s]
print(f"  r(their implied SC, vs wind) = {np.corrcoef(sc_implied[ok], w[ok])[0,1]:.3f}")
print(f"  r(our SC,           vs wind) = {np.corrcoef(sc_o_s[ok], w[ok])[0,1]:.3f}")
print("  If theirs is near zero, they are NOT driving SC with this wind field.")

print("\nSubstituting THEIR ERC into our BI (isolates the SC error)")
bi_hybrid = np.rint(3.01 * np.maximum(sc_o_s * ercg, 0) ** 0.46)
print(f"  r(BI) using our ERC + our SC:   {np.corrcoef(bi_o_s, big)[0,1]:.3f}")
print(f"  r(BI) using their ERC + our SC: {np.corrcoef(bi_hybrid, big)[0,1]:.3f}")
print("  Little change confirms SC, not ERC, is the culprit.")

# --- the zero-BI days -------------------------------------------------
# SC correlates at ~0.97 and ERC at ~0.98, yet BI sits near 0.67. Since BI
# is a deterministic function of exactly those two, that cannot be right --
# and it is not. The SC comparison above is masked to (ercg>0)&(big>0), but
# the BI correlation is unmasked. gridMET reports BI of exactly 0 on some
# days while still reporting a non-zero ERC. Those days are in one
# comparison and not the other.
nzero = int(np.sum(~ok))
print(f"\nDays where gridMET reports BI = 0: {nzero} of {len(big)} "
      f"({100*nzero/len(big):.1f}%)")
if nzero:
    print(f"  their ERC on those days: mean {ercg[~ok].mean():.1f}")
    print(f"  OUR BI on those days:    mean {bi_o_s[~ok].mean():.1f}  "
          f"(we never shut spread off)")
    print(f"  precip on those days:    mean {ppt_in[s][~ok].mean():.3f} in")
    print(f"  our MC1 vs extinction:   these are the days SC should go to 0")
    print(f"\n  r(BI) all days:            {np.corrcoef(bi_o_s, big)[0,1]:.3f}")
    print(f"  r(BI) excluding BI=0 days: "
          f"{np.corrcoef(bi_o_s[ok], big[ok])[0,1]:.3f}")
    print(f"  bias excluding BI=0 days:  {np.mean(bi_o_s[ok]-big[ok]):+.2f}")
    print("\n  A large jump means BI is fine and the whole disagreement is")
    print("  these wet days: gridMET's fine fuels cross the 25 pct moisture")
    print("  of extinction and spread stops, ours never get that wet because")
    print("  we feed tmax/rmin as the 1300 LST observation -- the driest")
    print("  numbers in the day.")

print("\nWind sensitivity: does a different wind treatment help?")
print(f"{'treatment':<34}{'r(BI)':>8}{'bias':>9}{'mean SC':>10}")
trials = {
    "current (10m daily mean -> 20ft)": ws_base,
    "no height adjustment": vs[:m] * 2.23694,
    "constant 5 mph everywhere": np.full(m, 5.0),
    "gust proxy (1.6x daily mean)": ws_base * 1.6,
    "gust proxy (2.2x daily mean)": ws_base * 2.2,
}
for label, wind in trials.items():
    _, _, b = run(wind)
    bs = b[s]
    r = np.corrcoef(bs, big)[0, 1]
    print(f"{label:<34}{r:>8.3f}{np.mean(bs-big):>9.2f}"
          f"{np.mean(run(wind)[0][s]):>10.2f}")

print("\nReading this:")
print("  constant wind scores BEST -> gridMET's BI is not wind-driven the")
print("     way ours is; our wind input is injecting noise, not signal.")
print("  a gust multiplier wins    -> daily-mean wind is too smooth; NFDRS")
print("     wants the afternoon observation.")
print("  nothing helps much        -> the disagreement is in SC's moisture")
print("     terms (fine fuels), not in the wind at all.")
