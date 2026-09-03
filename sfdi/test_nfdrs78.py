"""Offline sanity checks for nfdrs78.

These cannot prove the engine correct -- only the gridMET comparison can do
that, and it needs network access we do not have here. What these DO catch is
the class of bug that produces plausible-looking output: wrong branch in a
piecewise function, a damping polynomial swapped between the SC and ERC
paths, state that fails to persist across days, or a sign error that makes
drought lower fire danger.
"""

import numpy as np
import nfdrs78 as n


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{' -- ' + detail if detail else ''}")
    return bool(cond)


results = []

# ---------------------------------------------------------------------------
print("\nEMC (Simard regressions, PSW-82 eq. 1a/1b/1c)")
# Branch boundaries are the classic place to get piecewise code wrong.
e_low = n.emc(80.0, 5.0)      # RH < 10 branch
e_mid = n.emc(80.0, 30.0)     # 10 <= RH < 50 branch
e_high = n.emc(80.0, 80.0)    # RH >= 50 branch
print(f"    EMC(80F,  5%) = {e_low:.2f}")
print(f"    EMC(80F, 30%) = {e_mid:.2f}")
print(f"    EMC(80F, 80%) = {e_high:.2f}")
results.append(check("EMC increases monotonically with RH", e_low < e_mid < e_high))
results.append(check("dry-end EMC is single digits", 0.0 < e_low < 5.0))
results.append(check("humid-end EMC is in the teens or twenties", 12.0 < e_high < 30.0))

# Hotter air at fixed RH holds fuels drier.
results.append(check("EMC decreases as temperature rises",
                     n.emc(100.0, 30.0) < n.emc(50.0, 30.0)))

# Continuity across the branch cuts -- a discontinuity here would inject
# artificial jumps into the fuel moisture recursion.
d10 = abs(n.emc(70.0, 9.999) - n.emc(70.0, 10.001))
d50 = abs(n.emc(70.0, 49.999) - n.emc(70.0, 50.001))
results.append(check("EMC roughly continuous at the RH=10 cut", d10 < 1.5, f"jump {d10:.3f}"))
results.append(check("EMC roughly continuous at the RH=50 cut", d50 < 1.5, f"jump {d50:.3f}"))

# ---------------------------------------------------------------------------
print("\nDaylength")
jun = n.daylength_hours(40.0, 172)   # summer solstice, 40N
dec = n.daylength_hours(40.0, 355)   # winter solstice, 40N
print(f"    40N, Jun 21 = {jun:.2f} h")
print(f"    40N, Dec 21 = {dec:.2f} h")
results.append(check("summer day near 15 h at 40N", 14.0 < jun < 15.5))
results.append(check("winter day near 9 h at 40N", 8.5 < dec < 10.0))
results.append(check("equator stays near 12 h",
                     abs(n.daylength_hours(0.0, 172) - 12.0) < 0.5))

# ---------------------------------------------------------------------------
print("\nFuel model G parameters")
fm = n.FUEL_G
results.append(check("1-h loading 2.5 t/ac", fm["W1"] == 2.5))
results.append(check("1000-h loading 12.0 t/ac", fm["W1000"] == 12.0))
results.append(check("dead moisture of extinction 25 pct", fm["MXD"] == 25.0))
results.append(check("wind reduction factor 0.4", fm["WNDFC"] == 0.4))

# ---------------------------------------------------------------------------
print("\nDrying run: 200 days of hot, dry, rainless weather")
N = 4
st = n.NFDRSState(N, climate_class=np.array([1.0, 2.0, 3.0, 4.0]))
erc_series, mc1000_series = [], []
for d in range(200):
    wx = dict(
        tmax_f=np.full(N, 95.0), tmin_f=np.full(N, 60.0),
        rhmax=np.full(N, 40.0), rhmin=np.full(N, 10.0),
        tobs_f=np.full(N, 93.0), rhobs=np.full(N, 12.0),
        sow=np.zeros(N, dtype=np.int8), ppt_in=np.zeros(N),
        ws_mph=np.full(N, 8.0), lat=np.full(N, 40.0), jday=150 + d % 200,
    )
    out = n.step(st, wx)
    erc_series.append(out["erc"].copy())
    mc1000_series.append(out["mc1000"].copy())

erc_series = np.array(erc_series)
mc1000_series = np.array(mc1000_series)
print(f"    ERC    day 1 / 30 / 200 : {erc_series[0]} / {erc_series[29]} / {erc_series[-1]}")
print(f"    MC1000 day 1 / 30 / 200 : {np.round(mc1000_series[0],1)} / "
      f"{np.round(mc1000_series[29],1)} / {np.round(mc1000_series[-1],1)}")

results.append(check("1000-h fuels dry out over a rainless season",
                     np.all(mc1000_series[-1] < mc1000_series[0])))
results.append(check("1000-h moisture stays physical (0-60 pct)",
                     np.all(mc1000_series > 0) and np.all(mc1000_series < 60)))
results.append(check("ERC rises as the season dries",
                     np.all(erc_series[-1] >= erc_series[0])))
results.append(check("ERC lands in a plausible Fuel Model G range",
                     np.all(erc_series[-1] > 20) and np.all(erc_series[-1] < 130),
                     f"final {erc_series[-1]}"))

# ---------------------------------------------------------------------------
print("\nWetting run: soak the dried-out fuels")
erc_before = erc_series[-1].copy()
for d in range(20):
    wx = dict(
        tmax_f=np.full(N, 60.0), tmin_f=np.full(N, 45.0),
        rhmax=np.full(N, 95.0), rhmin=np.full(N, 70.0),
        tobs_f=np.full(N, 58.0), rhobs=np.full(N, 75.0),
        sow=np.full(N, 3, dtype=np.int8), ppt_in=np.full(N, 0.5),
        ws_mph=np.full(N, 4.0), lat=np.full(N, 40.0), jday=250 + d,
    )
    out = n.step(st, wx)
# The wet-fuel rule is two behaviors and gridMET uses one without the other:
# fine fuels pinned above the moisture of extinction (so SC and BI go to 0
# through the physics) while ERC survives on the 100/1000-h fuels. gridMET's
# ERC bottoms out at 3, never 0, and 11.4% of their days have BI exactly 0.
st_w = n.NFDRSState(1, climate_class=np.array([1.0]))
for _ in range(300):
    n.step(st_w, dict(
        tmax_f=np.full(1, 90.0), tmin_f=np.full(1, 58.0),
        rhmax=np.full(1, 45.0), rhmin=np.full(1, 14.0),
        tobs_f=np.full(1, 90.0), rhobs=np.full(1, 14.0),
        sow=np.zeros(1, dtype=np.int8), ppt_in=np.zeros(1),
        ws_mph=np.full(1, 7.0), lat=np.full(1, 39.5), jday=180))
o_wet = n.step(st_w, dict(
    tmax_f=np.full(1, 70.0), tmin_f=np.full(1, 50.0),
    rhmax=np.full(1, 85.0), rhmin=np.full(1, 45.0),
    tobs_f=np.full(1, 70.0), rhobs=np.full(1, 45.0),
    sow=np.full(1, 3, dtype=np.int8), ppt_in=np.full(1, 0.2),
    ws_mph=np.full(1, 7.0), lat=np.full(1, 39.5), jday=181))
results.append(check("wet day drives SC to zero", o_wet["sc"][0] == 0))
results.append(check("wet day drives BI to zero", o_wet["bi"][0] == 0))
results.append(check("wet day leaves ERC standing", o_wet["erc"][0] > 20,
                     f"ERC {o_wet['erc'][0]:.0f}"))
st_z = n.NFDRSState(1, climate_class=np.array([1.0]), zero_components=True)
o_z = n.step(st_z, dict(
    tmax_f=np.full(1, 70.0), tmin_f=np.full(1, 50.0),
    rhmax=np.full(1, 85.0), rhmin=np.full(1, 45.0),
    tobs_f=np.full(1, 70.0), rhobs=np.full(1, 45.0),
    sow=np.full(1, 3, dtype=np.int8), ppt_in=np.full(1, 0.2),
    ws_mph=np.full(1, 7.0), lat=np.full(1, 39.5), jday=181))
results.append(check("station-mode zeroing still available on request",
                     o_z["erc"][0] == 0))
results.append(check("1000-h fuels recover after sustained rain",
                     np.all(out["mc1000"] > mc1000_series[-1])))

# Dry it back out and confirm ERC comes back -- proves state is persisting
# rather than being silently reset each call.
for d in range(60):
    wx = dict(
        tmax_f=np.full(N, 95.0), tmin_f=np.full(N, 60.0),
        rhmax=np.full(N, 40.0), rhmin=np.full(N, 10.0),
        tobs_f=np.full(N, 93.0), rhobs=np.full(N, 12.0),
        sow=np.zeros(N, dtype=np.int8), ppt_in=np.zeros(N),
        ws_mph=np.full(N, 8.0), lat=np.full(N, 40.0), jday=280 + d,
    )
    out = n.step(st, wx)
results.append(check("ERC recovers after the fuels dry again",
                     np.all(out["erc"] > 0.5 * erc_before)))

# ---------------------------------------------------------------------------
print("\nSeparating ERC from BI (the two must not move together)")
# ERC is a build-up index and should barely notice wind. BI folds in spread,
# so it must. If ERC responds to wind, the weighting schemes got crossed.
st_a = n.NFDRSState(1, climate_class=np.array([2.0]))
st_b = n.NFDRSState(1, climate_class=np.array([2.0]))
for d in range(120):
    base = dict(
        tmax_f=np.full(1, 92.0), tmin_f=np.full(1, 58.0),
        rhmax=np.full(1, 45.0), rhmin=np.full(1, 12.0),
        tobs_f=np.full(1, 90.0), rhobs=np.full(1, 14.0),
        sow=np.zeros(1, dtype=np.int8), ppt_in=np.zeros(1),
        lat=np.full(1, 40.0), jday=150 + d,
    )
    calm = n.step(st_a, dict(base, ws_mph=np.full(1, 2.0)))
    windy = n.step(st_b, dict(base, ws_mph=np.full(1, 25.0)))

print(f"    calm  (2 mph): ERC={calm['erc'][0]:.0f}  BI={calm['bi'][0]:.0f}")
print(f"    windy (25 mph): ERC={windy['erc'][0]:.0f}  BI={windy['bi'][0]:.0f}")
results.append(check("ERC is insensitive to wind",
                     abs(calm["erc"][0] - windy["erc"][0]) < 1.0))
results.append(check("BI rises sharply with wind",
                     windy["bi"][0] > 1.5 * max(calm["bi"][0], 1.0)))

# ---------------------------------------------------------------------------
# X1000 drives the live fuel models. It is a DAMPED follower of MC1000, and
# an earlier version accumulated daily increments -- which ratcheted downward
# about 7 units per year because drying is undamped while wetting is not.
# That pinned herbaceous moisture at its clip floor and biased ERC high at
# every site. Guard against the regression.
print("\nX1000 must track MC1000 without drifting")
st_dr = n.NFDRSState(1, climate_class=np.array([1.0]))
gaps, herb_seen = [], []
for d in range(6 * 365):
    doy = d % 365
    seas = np.sin((doy - 100) / 365 * 2 * np.pi)
    o = n.step(st_dr, dict(
        tmax_f=np.full(1, 70 + 22 * seas), tmin_f=np.full(1, 40 + 18 * seas),
        rhmax=np.full(1, 75 - 25 * seas), rhmin=np.full(1, 30 - 20 * seas),
        tobs_f=np.full(1, 70 + 22 * seas), rhobs=np.full(1, 30 - 20 * seas),
        sow=np.zeros(1, dtype=np.int8),
        ppt_in=np.full(1, 0.35 if (doy < 90 or doy > 300) and d % 4 == 0 else 0.0),
        ws_mph=np.full(1, 5.0), lat=np.full(1, 39.5), jday=doy + 1))
    if d > 365:
        gaps.append(abs(st_dr.x1000[0] - o["mc1000"][0]))
        herb_seen.append(o["mcherb"][0])
results.append(check("X1000 stays near MC1000 over six years",
                     max(gaps) < 15.0, f"max gap {max(gaps):.1f}"))
results.append(check("X1000 stays physical (never negative)",
                     st_dr.x1000[0] > 0, f"final {st_dr.x1000[0]:.1f}"))
results.append(check("herbaceous moisture actually varies seasonally",
                     max(herb_seen) - min(herb_seen) > 40,
                     f"range {min(herb_seen):.0f}-{max(herb_seen):.0f} pct"))

print("\nVectorization: many cells must equal one cell run separately")
M = 500
rng = np.random.default_rng(0)
clim = rng.integers(1, 5, M).astype(float)
st_many = n.NFDRSState(M, climate_class=clim)
st_one = n.NFDRSState(1, climate_class=clim[7:8])
tmax = 70.0 + rng.random((90, M)) * 30.0
rhmin = 5.0 + rng.random((90, M)) * 40.0
ppt = np.where(rng.random((90, M)) < 0.1, 0.3, 0.0)
for d in range(90):
    common = dict(sow=np.zeros(M, dtype=np.int8), lat=np.full(M, 42.0), jday=120 + d)
    out_m = n.step(st_many, dict(
        tmax_f=tmax[d], tmin_f=tmax[d] - 30.0, rhmax=rhmin[d] + 40.0,
        rhmin=rhmin[d], tobs_f=tmax[d] - 2.0, rhobs=rhmin[d] + 2.0,
        ppt_in=ppt[d], ws_mph=np.full(M, 7.0), **common))
    out_1 = n.step(st_one, dict(
        tmax_f=tmax[d, 7:8], tmin_f=tmax[d, 7:8] - 30.0,
        rhmax=rhmin[d, 7:8] + 40.0, rhmin=rhmin[d, 7:8],
        tobs_f=tmax[d, 7:8] - 2.0, rhobs=rhmin[d, 7:8] + 2.0,
        ppt_in=ppt[d, 7:8], ws_mph=np.full(1, 7.0),
        sow=np.zeros(1, dtype=np.int8), lat=np.full(1, 42.0), jday=120 + d))
results.append(check("cell 7 matches its standalone run",
                     np.isclose(out_m["erc"][7], out_1["erc"][0]),
                     f"{out_m['erc'][7]} vs {out_1['erc'][0]}"))
results.append(check("no NaN or inf anywhere in the field",
                     np.all(np.isfinite(out_m["erc"])) and np.all(np.isfinite(out_m["bi"]))))

# The single-day comparison above can pass degenerately: if cell 7 had rain
# that day, both sides are zeroed and the check proves nothing. Re-run the
# whole series, compare EVERY day, and require that a decent number of the
# matched days were actually non-zero.
st_many2 = n.NFDRSState(M, climate_class=clim)
st_one2 = n.NFDRSState(1, climate_class=clim[7:8])
diffs, nonzero_days = [], 0
for d in range(90):
    o_m = n.step(st_many2, dict(
        tmax_f=tmax[d], tmin_f=tmax[d] - 30.0, rhmax=rhmin[d] + 40.0,
        rhmin=rhmin[d], tobs_f=tmax[d] - 2.0, rhobs=rhmin[d] + 2.0,
        ppt_in=ppt[d], ws_mph=np.full(M, 7.0),
        sow=np.zeros(M, dtype=np.int8), lat=np.full(M, 42.0), jday=120 + d))
    o_1 = n.step(st_one2, dict(
        tmax_f=tmax[d, 7:8], tmin_f=tmax[d, 7:8] - 30.0,
        rhmax=rhmin[d, 7:8] + 40.0, rhmin=rhmin[d, 7:8],
        tobs_f=tmax[d, 7:8] - 2.0, rhobs=rhmin[d, 7:8] + 2.0,
        ppt_in=ppt[d, 7:8], ws_mph=np.full(1, 7.0),
        sow=np.zeros(1, dtype=np.int8), lat=np.full(1, 42.0), jday=120 + d))
    diffs.append(abs(o_m["erc"][7] - o_1["erc"][0]))
    if o_1["erc"][0] > 0:
        nonzero_days += 1
results.append(check("every day of the series matches", max(diffs) < 1e-9,
                     f"max diff {max(diffs):.2e}"))
results.append(check("the comparison was not vacuous", nonzero_days > 40,
                     f"{nonzero_days}/90 days had non-zero ERC"))

# Neighbouring cells with different weather must not equal each other --
# catches accidental broadcasting of a scalar across the whole field.
spread = np.std(o_m["erc"])
results.append(check("cells differ from one another", spread > 1.0,
                     f"ERC std across {M} cells = {spread:.1f}"))

# ---------------------------------------------------------------------------
print("\nThroughput estimate for the climatology build")
import time
BIG = 240_000
st_big = n.NFDRSState(BIG, climate_class=np.full(BIG, 2.0))
wx_big = dict(
    tmax_f=np.full(BIG, 90.0), tmin_f=np.full(BIG, 55.0),
    rhmax=np.full(BIG, 50.0), rhmin=np.full(BIG, 15.0),
    tobs_f=np.full(BIG, 88.0), rhobs=np.full(BIG, 17.0),
    sow=np.zeros(BIG, dtype=np.int8), ppt_in=np.zeros(BIG),
    ws_mph=np.full(BIG, 8.0), lat=np.full(BIG, 42.0), jday=200,
)
t0 = time.time()
for _ in range(10):
    n.step(st_big, wx_big)
per_day = (time.time() - t0) / 10
print(f"    {per_day*1000:.0f} ms per day at {BIG:,} cells")
print(f"    30-yr climatology (~10,958 days): {per_day*10958/60:.1f} minutes of compute")
results.append(check("30-yr run finishes in well under an hour",
                     per_day * 10958 < 3600))

print(f"\n{sum(results)}/{len(results)} passed")
