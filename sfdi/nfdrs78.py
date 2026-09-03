"""
NFDRS 1978/88 engine -- ERC, BI and SC for a grid of cells.

Implements Cohen & Deeming (1985), "The National Fire-Danger Rating System:
basic equations", USDA Forest Service Gen. Tech. Rep. PSW-82. Equation
numbering in comments refers to that document.

Design notes
------------
* Vectorized ACROSS CELLS, sequential ACROSS DAYS. The dead-fuel moisture
  models are recursive in time (MC100 depends on yesterday, MC1000 on a
  7-day running mean), so days cannot be parallelized. Cells are fully
  independent, so every operation below is a numpy array op over an
  (n_cells,) vector. For the western US at 1/24 degree that is ~240k
  cells per day, which numpy handles in milliseconds.

* Fuel Model G is hardwired. Standardizing on one fuel model is the
  accepted convention when comparing fire danger across large areas
  (it isolates weather variation from fuel variation), it is what the
  SFDI paper used, and it means we need no fuel model map.

* Everything here is deterministic given (weather, state). There is no
  fitting, no tuning. That matters because the whole point of choosing
  the legacy system was that gridMET publishes ERC/BI computed the same
  way, giving us an independent answer key to validate against.

UNITS -- the 1978 system is imperial and mixing units up is the classic
way to get plausible-looking wrong answers:
    temperature     degrees F  (except QIGN, which wants Celsius)
    wind            mph, 20-ft height
    fuel loading    tons/acre on input, converted to lb/ft2 internally
    fuel moisture   percent of dry weight
    SAV ratio       ft^-1
"""

import numpy as np

# --------------------------------------------------------------------------
# Fuel Model G -- "short needle, heavy dead" conifer.
# From the PSW-82 appendix. Loadings are tons/acre, SAV ratios ft^-1.
# --------------------------------------------------------------------------
FUEL_G = dict(
    W1=2.5, W10=2.0, W100=5.0, W1000=12.0,   # dead loadings, tons/acre
    WHERB=0.5, WWOOD=0.5,                     # live loadings, tons/acre
    SG1=2000.0, SG10=109.0, SG100=30.0, SG1000=8.0,
    SGHERB=2000.0, SGWOOD=1500.0,
    DEPTH=1.0,        # effective fuel bed depth, ft
    MXD=25.0,         # dead fuel moisture of extinction, percent
    HD=8000.0,        # dead fuel heat of combustion, Btu/lb
    HL=8000.0,        # live fuel heat of combustion, Btu/lb
    SCM=30.0,         # SC at which all ignitions become reportable
    WNDFC=0.4,        # wind reduction factor, 20-ft to midflame
    HERB_ANNUAL=False,  # Fuel Model G herbaceous component is perennial
)

TONS_ACRE_TO_LB_FT2 = 0.0459137   # PSW-82, "Preliminary Calculations"
STD = STL = 0.0555   # inert mineral fraction, dead and live
SD = SL = 0.01       # silica-free mineral fraction, dead and live
RHOD = RHOL = 32.0   # particle density, lb/ft3


# --------------------------------------------------------------------------
# Equilibrium moisture content -- Simard (1968), PSW-82 eq. 1a/1b/1c.
# --------------------------------------------------------------------------
def emc(temp_f, rh):
    """Equilibrium moisture content (percent) from dry-bulb F and RH percent.

    Three-branch piecewise regression. np.select evaluates all branches, so
    guard the inputs rather than the expressions -- RH of exactly 0 or 100
    is legal here, but NaN propagates and will silently poison the whole
    downstream recursion, so clip instead of letting it through.
    """
    temp_f = np.asarray(temp_f, dtype=np.float64)
    rh = np.clip(np.asarray(rh, dtype=np.float64), 0.0, 100.0)

    below10 = 0.03229 + 0.281073 * rh - 0.000578 * temp_f * rh
    mid = 2.22749 + 0.160107 * rh - 0.014784 * temp_f
    above50 = (21.0606 + 0.005565 * rh ** 2
               - 0.00035 * rh * temp_f - 0.483199 * rh)

    out = np.select([rh < 10.0, rh < 50.0], [below10, mid], default=above50)
    # The regressions can go slightly negative at extreme hot/dry combinations.
    # Physically EMC cannot be below zero and NFDRS treats it as bounded.
    return np.clip(out, 0.0, 100.0)


def daylength_hours(lat_deg, jday):
    """Hours between sunrise and sunset -- PSW-82, "Duration of Daylight".

    Used to weight EMCMIN (day) against EMCMAX (night) for the 100- and
    1000-hour fuels. The arccos argument runs out of domain above the polar
    circles; clipping yields 24h/0h polar day/night, which is correct
    behavior and irrelevant for the western US anyway.
    """
    phi = np.asarray(lat_deg, dtype=np.float64) * 0.01745
    decl = 0.41008 * np.sin((jday - 82) * 0.01745)
    x = np.clip(np.tan(phi) * np.tan(decl), -1.0, 1.0)
    return 24.0 * (1.0 - np.arccos(x) / np.pi)


# --------------------------------------------------------------------------
# Insolation correction for the 1-hour fuels.
# PSW-82: temperature correction is ADDED (F), RH correction is a MULTIPLIER,
# both keyed to the state-of-weather code.
#   0 = clear, 1 = scattered, 2 = broken, 3 = overcast
# --------------------------------------------------------------------------
_SOW_TEMP_ADD = np.array([25.0, 19.0, 12.0, 5.0])
_SOW_RH_MULT = np.array([0.75, 0.83, 0.92, 1.00])


def state_of_weather_from_srad(srad, clear_sky_srad):
    """Derive a 0-3 state-of-weather code from a solar radiation ratio.

    NFDRS expects a human observer's sky-cover code. Gridded input has no
    observer, so we bin the ratio of actual to clear-sky shortwave. The bin
    edges below are a modeling choice, NOT something PSW-82 specifies --
    this is one of the few places our implementation could legitimately
    diverge from gridMET's, and therefore one of the first places to look
    if validation shows a systematic ERC offset.
    """
    ratio = np.divide(srad, np.maximum(clear_sky_srad, 1e-6))
    return np.select(
        [ratio >= 0.85, ratio >= 0.60, ratio >= 0.35],
        [0, 1, 2],
        default=3,
    ).astype(np.int8)


class NFDRSState:
    """Carries the recursive fuel-moisture state between days.

    This object IS the thing that makes forecasting possible. ERC is a
    build-up index: MC1000 has weeks of memory, so a forecast run cannot
    start cold. It must be initialized by integrating observed weather
    forward, then handed to the forecast weather. Persist this between
    runs.
    """

    def __init__(self, n_cells, climate_class, herb_annual=False,
                 wet_fuel_pinning=True, zero_components=False,
                 wet_threshold_in=0.01):
        self.n = n_cells
        # The NFDRS wet-fuel rule is TWO separate behaviors, and gridMET
        # applies one without the other. Bundling them was wrong.
        #
        #   wet_fuel_pinning   -- fine fuels pinned to 35 pct on wet days.
        #       35 exceeds Fuel Model G's 25 pct moisture of extinction, so
        #       the dead-fuel damping term goes to zero and SPREAD STOPS:
        #       SC and therefore BI fall to 0. ERC survives, because its
        #       loading weighting is dominated by the 100- and 1000-hour
        #       fuels, which stay dry. gridMET does this -- 11.4 pct of days
        #       at Reno have BI exactly 0 while ERC still averages 36.
        #
        #   zero_components    -- SC, ERC and BI all forced to 0. This is the
        #       operational station convention and gridMET does NOT do it
        #       (their ERC bottoms out at 3, never 0). Leave it off for
        #       gridded work.
        #
        # Defaults match gridMET so the climatology and the daily product
        # stay consistent with the reference we validated against.
        self.wet_fuel_pinning = wet_fuel_pinning
        self.zero_components = zero_components
        self.wet_threshold_in = wet_threshold_in
        # NFDRS climate class 1-4 (1 = arid, 4 = wet). Controls greenup
        # length, dormant woody moisture, the herbaceous regressions, and
        # the assumed rainfall rate.
        self.climat = np.asarray(climate_class, dtype=np.float64)
        self.herb_annual = herb_annual

        # PSW-82 "Initializing YMC100 / MC1000 / BNDRYT at the beginning of
        # a computational period". These are deliberately crude; they are
        # why a spin-up period is mandatory before output is trustworthy.
        self.mc100 = 5.0 + 5.0 * self.climat
        self.mc1000 = np.full((7, n_cells), 10.0 + 5.0 * self.climat)
        self.bndryt = np.full((7, n_cells), 10.0 + 5.0 * self.climat)
        self.mc10 = np.full(n_cells, 10.0)

        self.x1000 = self.mc1000[-1].copy()
        self.mcherb = np.full(n_cells, 30.0)
        self.mcwood = self._pregrn()
        self.grnday = np.zeros(n_cells, dtype=np.int32)
        self.greenup = np.zeros(n_cells, dtype=bool)
        self.day_count = 0

    def _pregrn(self):
        """Dormant woody moisture: 50/60/70/80 pct for climate class 1/2/3/4."""
        return 40.0 + 10.0 * self.climat

    def wetrat(self):
        """Assumed rainfall rate, in/hr, used to turn amount into duration."""
        return np.where(self.climat <= 2, 0.25, 0.05)


def _ppt_duration(ppt_in, wetrat):
    """Pseudo rain duration in hours from precip amount -- PSW-82.

    PPTDUR = IRND(PPTAMT / WETRAT + 0.49), capped at 8 hours. Gridded daily
    precip has no duration, so this conversion is unavoidable.
    """
    dur = np.rint(ppt_in / wetrat + 0.49)
    return np.clip(dur, 0.0, 8.0)


def _update_dead_moisture(st, tmax_f, tmin_f, rhmax, rhmin,
                          tobs_f, rhobs, sow, ppt_in, lat, jday):
    """Advance MC1, MC10, MC100, MC1000 by one day."""
    # --- 1-hour: EMC at the fuel/atmosphere interface, insolation-corrected
    tmpprm = tobs_f + _SOW_TEMP_ADD[sow]
    rhprm = rhobs * _SOW_RH_MULT[sow]
    emcprm = emc(tmpprm, rhprm)
    mc1 = 1.03 * emcprm            # no fuel-moisture sticks in a gridded run

    # --- 10-hour: observation form (sticks not used)
    mc10 = 1.28 * emcprm

    # --- 24-hour average EMC, weighted by day and night hours
    emcmin = emc(tmax_f, rhmin)    # hot and dry -> daytime
    emcmax = emc(tmin_f, rhmax)    # cool and moist -> nighttime
    daylit = daylength_hours(lat, jday)
    emcbar = (daylit * emcmin + (24.0 - daylit) * emcmax) / 24.0

    pptdur = _ppt_duration(ppt_in, st.wetrat())

    # --- 100-hour
    bndryh = ((24.0 - pptdur) * emcbar + pptdur * (0.5 * pptdur + 41.0)) / 24.0
    mc100 = st.mc100 + (bndryh - st.mc100) * (1.0 - 0.87 * np.exp(-0.24))

    # --- 1000-hour: 7-day running mean of the boundary condition, and the
    # update is against the value from SEVEN days ago, not yesterday.
    bndryt = ((24.0 - pptdur) * emcbar + pptdur * (2.7 * pptdur + 76.0)) / 24.0
    st.bndryt = np.roll(st.bndryt, -1, axis=0)
    st.bndryt[-1] = bndryt
    bdybar = st.bndryt.mean(axis=0)

    pm1000 = st.mc1000[0]          # oldest entry == seventh previous day
    mc1000 = pm1000 + (bdybar - pm1000) * (1.0 - 0.82 * np.exp(-0.168))
    st.mc1000 = np.roll(st.mc1000, -1, axis=0)
    st.mc1000[-1] = mc1000

    # --- wet-fuel rule: raining at observation time pins the fine fuels.
    # Only applies when we have an actual observation-time rain flag. Daily
    # gridded input does not: knowing it rained somewhere in a 24-hour window
    # says nothing about conditions at 1300 LST. gridMET's published ERC has
    # a floor of 5 over four years at Reno, confirming they never zero.
    # The 100/1000-hour calculations continue uninterrupted either way.
    if st.wet_fuel_pinning:
        raining = ppt_in > st.wet_threshold_in
        mc1 = np.where(raining, 35.0, mc1)
        mc10 = np.where(raining, 35.0, mc10)

    st.mc100 = mc100
    st.mc10 = mc10
    return mc1, mc10, mc100, mc1000, tmpprm


def _update_live_moisture(st, mc1, mc1000, prev_mc1000, tmax_f, tmin_f):
    """Advance herbaceous and woody live fuel moisture by one day.

    This is the least mechanical part of the system. In operational NFDRS a
    human declares greenup and killing frost; a gridded run has to infer
    both. The rule used here is documented inline and is a second candidate
    explanation for any systematic divergence from gridMET.
    """
    climat = st.climat

    # X1000: a damped version of MC1000 that drives the live fuel models.
    #
    # KWET damps the response to WETTING (live fuels do not green up the
    # instant it rains) but leaves drying at full strength. Applied as an
    # accumulation of daily increments, that asymmetry ratchets: each wet-dry
    # cycle loses ground permanently, and X1000 drifted about -7 per year in
    # testing, reaching -30 after six years. The herbaceous regression then
    # returns large negatives, the clip catches them at 30 percent, and the
    # herb layer reads as permanently cured -- which showed up as a flat
    # positive ERC bias at every site.
    #
    # Written as a relaxation toward MC1000 instead. A damped follower can
    # lag, which is the intended behavior, but it cannot drift away.
    kwet = np.select(
        [mc1000 > 25.0, mc1000 >= 10.0],
        [1.0, 0.0333 * mc1000 + 0.1675],
        default=0.5,
    )
    kwet = np.where(mc1000 >= st.x1000, kwet, 1.0)   # drying is never damped
    ktmp = np.where((tmax_f + tmin_f) / 2.0 <= 50.0, 0.6, 1.0)
    st.x1000 = st.x1000 + (mc1000 - st.x1000) * kwet * ktmp
    # Guard: X1000 stands in for a fuel moisture and must stay physical.
    st.x1000 = np.clip(st.x1000, 1.0, 60.0)

    # --- automatic greenup / freeze detection ------------------------------
    # PSW-82 leaves this to the user. We start greenup when the 1000-hour
    # fuels are still moist and it has warmed up, and force dormancy on a
    # hard freeze, following the FIREFAMILY convention of a killing frost at
    # TMIN <= 25 F.
    warm = (tmax_f + tmin_f) / 2.0 > 50.0
    killing_frost = tmin_f <= 25.0
    start = (~st.greenup) & warm & (mc1000 > 10.0 + 2.0 * climat)
    st.greenup = np.where(start, True, st.greenup)
    st.grnday = np.where(start, 0, st.grnday + 1)
    st.greenup = np.where(killing_frost, False, st.greenup)

    gren = np.clip(st.grnday / (7.0 * climat), 0.0, 1.0)

    # Herbaceous potential moisture, linear in X1000 with climate-class
    # coefficients (PSW-82 HERBGA/HERBGB table).
    herbga = np.select([climat == 1, climat == 2, climat == 3],
                       [-70.0, -100.0, -137.5], default=-185.0)
    herbgb = np.select([climat == 1, climat == 2, climat == 3],
                       [12.8, 14.0, 15.5], default=17.4)
    mchrbp = herbga + herbgb * st.x1000

    if st.herb_annual:
        anta = np.select([climat == 1, climat == 2, climat == 3],
                         [-150.5, -187.7, -245.2], default=-305.2)
        antb = np.select([climat == 1, climat == 2, climat == 3],
                         [18.4, 19.6, 22.0], default=24.3)
        mcherb_trans = anta + antb * st.x1000
    else:
        perta = np.select([climat == 1, climat == 2, climat == 3],
                          [11.2, -10.3, -42.7], default=-93.5)
        pertb = np.select([climat == 1, climat == 2, climat == 3],
                          [7.4, 8.3, 9.8], default=12.2)
        mcherb_trans = perta + pertb * st.x1000

    mcherb = np.where(gren < 1.0,
                      st.mcherb + (mchrbp - st.mcherb) * gren,
                      np.where(mchrbp > 120.0, mchrbp, mcherb_trans))
    mcherb = np.where(st.greenup, mcherb, mc1)   # cured -> tracks 1-hour fuels
    if st.herb_annual:
        # Annuals may not re-wet once curing has begun.
        mcherb = np.minimum(mcherb, st.mcherb)
    mcherb = np.clip(mcherb, 30.0, 250.0)
    st.mcherb = mcherb

    # Woody shrubs: simpler, no loading transfer, four stages.
    woodga = np.select([climat == 1, climat == 2, climat == 3],
                       [12.5, -5.0, -22.5], default=-45.0)
    woodgb = np.select([climat == 1, climat == 2, climat == 3],
                       [7.5, 8.2, 8.9], default=9.8)
    pregrn = st._pregrn()
    mcwodp = woodga + woodgb * mc1000
    mcwood = np.where(gren < 1.0,
                      st.mcwood + (mcwodp - st.mcwood) * gren,
                      mcwodp)
    mcwood = np.where(st.greenup, mcwood, pregrn)
    mcwood = np.clip(mcwood, pregrn, 200.0)
    st.mcwood = mcwood

    return mcherb, mcwood


def _components(mc1, mc10, mc100, mc1000, mcherb, mcwood, ws_mph, fm):
    """Spread Component, Energy Release Component and Burning Index.

    Two parallel weighting schemes run here and they are easy to conflate:
    SC weights fuel classes by SURFACE AREA, ERC weights them by LOADING.
    That is why ERC responds to the big 1000-hour fuels (drought) while SC
    responds to the fine fuels and wind. They also use DIFFERENT moisture
    damping polynomials.
    """
    # --- herbaceous loading transfer, PSW-82 eq. 5-8
    fctcur = np.clip(1.33 - 0.0111 * mcherb, 0.0, 1.0)
    wherbc = fctcur * fm["WHERB"]
    w1p = (fm["W1"] + wherbc) * TONS_ACRE_TO_LB_FT2
    wherbp = (fm["WHERB"] - wherbc) * TONS_ACRE_TO_LB_FT2

    w10 = fm["W10"] * TONS_ACRE_TO_LB_FT2
    w100 = fm["W100"] * TONS_ACRE_TO_LB_FT2
    w1000 = fm["W1000"] * TONS_ACRE_TO_LB_FT2
    wwood = fm["WWOOD"] * TONS_ACRE_TO_LB_FT2

    wtotd = w1p + w10 + w100 + w1000
    wtotl = wherbp + wwood
    wtot = wtotd + wtotl

    rhobed = (wtot - w1000) / fm["DEPTH"]
    rhobar = (wtotl * RHOL + wtotd * RHOD) / wtot
    betbar = rhobed / rhobar

    etasd = 0.174 * SD ** -0.19
    etasl = 0.174 * SL ** -0.19

    # Net (mineral-free) loadings
    w1n = w1p * (1.0 - STD)
    w10n = w10 * (1.0 - STD)
    w100n = w100 * (1.0 - STD)
    wherbn = wherbp * (1.0 - STL)
    wwoodn = wwood * (1.0 - STL)

    # Heating numbers. Note the 1000-hour class is deliberately omitted --
    # its SAV is so low its contribution is negligible (PSW-82).
    hn1 = w1n * np.exp(-138.0 / fm["SG1"])
    hn10 = w10n * np.exp(-138.0 / fm["SG10"])
    hn100 = w100n * np.exp(-138.0 / fm["SG100"])
    hnherb = wherbn * np.exp(-500.0 / fm["SGHERB"])
    hnwood = wwoodn * np.exp(-500.0 / fm["SGWOOD"])
    wrat = (hn1 + hn10 + hn100) / np.maximum(hnherb + hnwood, 1e-12)

    # ---------------- surface-area weighting (Spread Component) -----------
    sa1 = (w1p / RHOD) * fm["SG1"]
    sa10 = (w10 / RHOD) * fm["SG10"]
    sa100 = (w100 / RHOD) * fm["SG100"]
    saherb = (wherbp / RHOL) * fm["SGHERB"]
    sawood = (wwood / RHOL) * fm["SGWOOD"]
    sadead = sa1 + sa10 + sa100
    salive = np.maximum(saherb + sawood, 1e-12)

    f1, f10, f100 = sa1 / sadead, sa10 / sadead, sa100 / sadead
    fherb, fwood = saherb / salive, sawood / salive
    fdead = sadead / (sadead + salive)
    flive = salive / (sadead + salive)

    wdeadn = f1 * w1n + f10 * w10n + f100 * w100n
    wliven = fwood * wwoodn + fherb * wherbn
    sgbrd = f1 * fm["SG1"] + f10 * fm["SG10"] + f100 * fm["SG100"]
    sgbrl = fherb * fm["SGHERB"] + fwood * fm["SGWOOD"]
    sgbrt = fdead * sgbrd + flive * sgbrl

    betop = 3.348 * sgbrt ** -0.8189
    gmamx = sgbrt ** 1.5 / (495.0 + 0.0594 * sgbrt ** 1.5)
    ad = 133.0 * sgbrt ** -0.7913
    gmaop = gmamx * (betbar / betop) ** ad * np.exp(ad * (1.0 - betbar / betop))
    zeta = (np.exp((0.792 + 0.681 * sgbrt ** 0.5) * (betbar + 0.1))
            / (192.0 + 0.2595 * sgbrt))

    # Live moisture of extinction depends on how dry the dead fuels are.
    mclfe = (mc1 * hn1 + mc10 * hn10 + mc100 * hn100) / np.maximum(
        hn1 + hn10 + hn100, 1e-12)
    mxl = np.maximum((2.9 * wrat * (1.0 - mclfe / fm["MXD"]) - 0.226) * 100.0,
                     fm["MXD"])

    wtmcd = f1 * mc1 + f10 * mc10 + f100 * mc100
    wtmcl = fherb * mcherb + fwood * mcwood
    dedrt = wtmcd / fm["MXD"]
    livrt = wtmcl / mxl
    etamd = np.clip(1.0 - 2.59 * dedrt + 5.11 * dedrt ** 2 - 3.52 * dedrt ** 3,
                    0.0, 1.0)
    etaml = np.clip(1.0 - 2.59 * livrt + 5.11 * livrt ** 2 - 3.52 * livrt ** 3,
                    0.0, 1.0)

    ir = gmaop * (wdeadn * fm["HD"] * etasd * etamd
                  + wliven * fm["HL"] * etasl * etaml)

    b = 0.02526 * sgbrt ** 0.54
    c = 7.47 * np.exp(-0.133 * sgbrt ** 0.55)
    e = 0.715 * np.exp(-3.59e-4 * sgbrt)
    ufact = c * (betbar / betop) ** -e
    # PSW-82 caps the wind term at 0.9 * IR.
    wind_term = np.minimum(ws_mph * 88.0 * fm["WNDFC"], 0.9 * ir)
    phiwnd = ufact * np.maximum(wind_term, 0.0) ** b
    phislp = 0.0    # slope class 0 -- see module docstring on standardization

    htsink = rhobed * (
        fdead * (f1 * np.exp(-138.0 / fm["SG1"]) * (250.0 + 11.16 * mc1)
                 + f10 * np.exp(-138.0 / fm["SG10"]) * (250.0 + 11.16 * mc10)
                 + f100 * np.exp(-138.0 / fm["SG100"]) * (250.0 + 11.16 * mc100))
        + flive * (fherb * np.exp(-138.0 / fm["SGHERB"]) * (250.0 + 11.16 * mcherb)
                   + fwood * np.exp(-138.0 / fm["SGWOOD"]) * (250.0 + 11.16 * mcwood))
    )
    ros = ir * zeta * (1.0 + phislp + phiwnd) / np.maximum(htsink, 1e-12)
    sc = np.rint(ros)

    # ---------------- loading weighting (Energy Release Component) --------
    f1e, f10e = w1p / wtotd, w10 / wtotd
    f100e, f1000e = w100 / wtotd, w1000 / wtotd
    fherbe, fwoode = wherbp / np.maximum(wtotl, 1e-12), wwood / np.maximum(wtotl, 1e-12)
    fdeade, flivee = wtotd / wtot, wtotl / wtot

    wdedne = wtotd * (1.0 - STD)
    wlivne = wtotl * (1.0 - STL)
    sgbrde = (f1e * fm["SG1"] + f10e * fm["SG10"]
              + f100e * fm["SG100"] + f1000e * fm["SG1000"])
    sgbrle = fwoode * fm["SGWOOD"] + fherbe * fm["SGHERB"]
    sgbrte = fdeade * sgbrde + flivee * sgbrle

    betope = 3.348 * sgbrte ** -0.8189
    gmamxe = sgbrte ** 1.5 / (495.0 + 0.0594 * sgbrte ** 1.5)
    ade = 133.0 * sgbrte ** -0.7913
    gmaope = (gmamxe * (betbar / betope) ** ade
              * np.exp(ade * (1.0 - betbar / betope)))

    wtmcde = f1e * mc1 + f10e * mc10 + f100e * mc100 + f1000e * mc1000
    wtmcle = fwoode * mcwood + fherbe * mcherb
    dedrte = wtmcde / fm["MXD"]
    livrte = wtmcle / mxl
    # NOTE the loading-weighted damping polynomial differs from the
    # surface-area one above. Using the SC polynomial here is a silent,
    # plausible-looking error.
    etamde = np.clip(1.0 - 2.0 * dedrte + 1.5 * dedrte ** 2 - 0.5 * dedrte ** 3,
                     0.0, 1.0)
    etamle = np.clip(1.0 - 2.0 * livrte + 1.5 * livrte ** 2 - 0.5 * livrte ** 3,
                     0.0, 1.0)

    ire = gmaope * (fdeade * wdedne * fm["HD"] * etasd * etamde
                    + flivee * wlivne * fm["HL"] * etasl * etamle)

    # Residence time uses the SURFACE-AREA weighted SAV, not the loading
    # weighted one. PSW-82 is explicit that the mass-weighted form gave
    # unrealistic results.
    tau = 384.0 / sgbrt
    erc = np.rint(0.04 * ire * tau)

    bi = np.rint(3.01 * np.maximum(sc * erc, 0.0) ** 0.46)
    return sc, erc, bi


def step(st, wx, fm=FUEL_G):
    """Advance one day. `wx` is a dict of same-shaped (n_cells,) arrays.

    Required keys: tmax_f, tmin_f, rhmax, rhmin, tobs_f, rhobs, sow,
    ppt_in, ws_mph, lat, jday.
    """
    prev_mc1000 = st.mc1000[-1].copy()

    mc1, mc10, mc100, mc1000, _ = _update_dead_moisture(
        st, wx["tmax_f"], wx["tmin_f"], wx["rhmax"], wx["rhmin"],
        wx["tobs_f"], wx["rhobs"], wx["sow"], wx["ppt_in"],
        wx["lat"], wx["jday"])

    mcherb, mcwood = _update_live_moisture(
        st, mc1, mc1000, prev_mc1000, wx["tmax_f"], wx["tmin_f"])

    sc, erc, bi = _components(mc1, mc10, mc100, mc1000,
                              mcherb, mcwood, wx["ws_mph"], fm)

    # Blanket zeroing of all three components is the operational station
    # convention, not gridMET's. Off by default -- with pinning enabled above,
    # SC and BI already fall to zero on wet days through the physics (fine
    # fuels above the moisture of extinction), while ERC correctly survives.
    if st.zero_components:
        wet = wx["ppt_in"] > st.wet_threshold_in
        sc = np.where(wet, 0.0, sc)
        erc = np.where(wet, 0.0, erc)
        bi = np.where(wet, 0.0, bi)

    st.day_count += 1
    return dict(sc=sc, erc=erc, bi=bi, mc1=mc1, mc10=mc10,
                mc100=mc100, mc1000=mc1000, mcherb=mcherb, mcwood=mcwood)
