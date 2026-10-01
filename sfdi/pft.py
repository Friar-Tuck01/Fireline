"""
pft.py — PyroCb Firepower Threshold (PFT), from scratch.

Implements the "simple PFT" of:

    Tory, K. J., and J. D. Kepert, 2021: Pyrocumulonimbus Firepower Threshold:
    Assessing the Atmospheric Potential for pyroCb. Wea. Forecasting, 36, 439-456.
    https://doi.org/10.1175/WAF-D-20-0027.1

PFT is the estimated MINIMUM firepower, in gigawatts, that a fire would need to
release in order to drive deep pyroconvection in a given atmospheric column.

    *** LOW PFT MEANS MORE FAVOURABLE FOR pyroCb, NOT LESS. ***

It is a threshold the atmosphere sets, not a hazard score. Any colour ramp built
on it must run "low = conducive". Getting this backwards inverts the product.

PFT says nothing about whether a particular fire will actually produce a pyroCb
-- that depends on the fire's own heat output, which this module does not model
and which satellite FRP does not directly measure (FRP is radiative power, a
~10-20% fraction of total heat release, sampled a few times a day and degraded
by the very smoke a pyroCb generates). Do not divide one by the other.

----------------------------------------------------------------------------
The equation
----------------------------------------------------------------------------
Paper Eq. (25), with the constants of Table 2 collapsed (Eq. 31):

    PFT(GW) = 0.3 * z_fc(km)^2 * U_ML(m/s) * dtheta_fc(K)

where the 0.3 is [pi*Cpd*(beta'/(1+alpha'*beta'))^2] * rho0 / 1000
                = 397.3 J/kg/K * 0.755 kg/m^3 / 1000 = 0.2999...

The arithmetic at the end is trivial. Everything hard is upstream, in getting
z_fc, dtheta_fc and U_ML out of a vertical profile.

----------------------------------------------------------------------------
The six-step procedure (paper section 4c)
----------------------------------------------------------------------------
1. theta_ML, q_ML : mass-weighted mean potential temperature and specific
   humidity from the surface to the mixed-layer LCL. Found iteratively: start
   shallow, deepen until the ML-LCL falls inside the ML estimate itself.
   Weighted linearly in z, because entrained mass flux grows linearly with
   plume height (paper Eqs. 15-16).

2. SP curve : a family of hypothetical plume parcels indexed by
   b = beta_T18 = dtheta_fc/theta_ML, running 0 -> ~0.1, with
       theta_SP = (b+1) * theta_ML                        (Eq. 29)
       q_SP     = q_ML + b * phi * theta_ML                (Eq. 30)
   phi = 6.67e-5 kg/kg/K, i.e. the fire adds 1 g/kg of moisture per 15 K of
   heating. Each SP is the saturation point (LCL) of that parcel.

3. z_fc : walk up the SP curve to the lowest parcel whose moist adiabat stays
   warmer than the environment all the way from its SP to the -20 C level (the
   conservative electrification level the paper adopts as minimum cloud top).
   Add the buoyancy buffer dtheta_b (0.5-1.0 K) to that minimum theta_e to
   allow for entrainment dilution. z_fc is the height of the resulting SP
   ABOVE THE FIRE GROUND.

4. dtheta_fc = theta_pl,fc - theta_ML = b_fc * theta_ML.

5. U_ML : magnitude of the VECTOR-mean wind (average u and v separately, then
   take the magnitude) from the surface to z_fc.

6. PFT from Eq. 31.

----------------------------------------------------------------------------
Two things that are easy to get wrong and fail quietly
----------------------------------------------------------------------------
* z_fc is height ABOVE THE FIRE, not above sea level and not above the sounding
  base. The paper subtracts 0.5 km for the Kinglake/Marysville fires and 100 m
  for Chisholm. z_fc is SQUARED, so a 1 km elevation error on a Sierra fire
  changes PFT by a factor of ~2-4. Pass fire_elev_m explicitly.
* Sign convention on the ramp: low PFT = conducive. See above.

----------------------------------------------------------------------------
Accuracy expectations, in the authors' own words
----------------------------------------------------------------------------
"PFT verification has been mainly qualitative. We expect that calculated PFT
values will contain both biases and random errors." They recommend comparing
values RELATIVELY (this fire vs that fire, this hour vs that hour) rather than
treating an absolute number as truth. Published anchor values are in
BENCHMARKS below and should be quoted alongside any displayed value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Constants (paper Table 2)
# ---------------------------------------------------------------------------
CPD = 1005.7          # J/kg/K   specific heat of dry air
RD = 287.04           # J/kg/K   gas constant for dry air
KAPPA = RD / CPD      # 0.2854
P0 = 1000.0           # hPa      reference pressure (Table 2 gives 1e5 Pa)
G = 9.8               # m/s^2
ALPHA_PRIME = 0.32    # fraction of plume radius below z_fc
BETA_PRIME = 0.40     # internal plume entrainment parameter
RHO0 = 0.755          # kg/m^3   fixed density used by the simple PFT
PHI = 6.67e-5         # kg/kg/K  fire heat-to-moisture ratio (15 K per 1 g/kg)
PHI_GKG = PHI * 1000.0  # 0.0667 g/kg/K, for working in g/kg

# Bracketed constant of Eq. 25, [pi*Cpd*(beta'/(1+alpha'*beta'))^2]
PFT_BRACKET = math.pi * CPD * (BETA_PRIME / (1.0 + ALPHA_PRIME * BETA_PRIME)) ** 2  # 397.3 J/kg/K
# Collapsed coefficient of Eq. 31, with z in km and PFT in GW
PFT_COEFF = PFT_BRACKET * RHO0 / 1000.0  # ~= 0.2999

# Algorithm settings
CLOUD_TOP_C = -20.0   # electrification level used as minimum cloud top
DTHETA_B = 0.5        # K, buoyancy buffer (paper: "~0.5-1.0", arbitrary)
B_MAX = 0.12          # upper bound on beta_T18 search (paper says "to about 0.1")

EPS = 0.622           # Rd/Rv


# ---------------------------------------------------------------------------
# Thermodynamics
#
# Written out rather than pulled from MetPy so every formula used is visible
# and testable. Each is checked in test_pft.py against published values.
# ---------------------------------------------------------------------------
def sat_vapor_pressure(T_c: float) -> float:
    """Saturation vapour pressure over liquid water, hPa. Bolton (1980) Eq. 10."""
    return 6.112 * math.exp(17.67 * T_c / (T_c + 243.5))


def q_from_e(e_hpa: float, p_hpa: float) -> float:
    """Specific humidity (g/kg) from vapour pressure and pressure."""
    return 1000.0 * EPS * e_hpa / (p_hpa - (1.0 - EPS) * e_hpa)


def e_from_q(q_gkg: float, p_hpa: float) -> float:
    """Vapour pressure (hPa) from specific humidity (g/kg) and pressure."""
    q = q_gkg / 1000.0
    return q * p_hpa / (EPS + (1.0 - EPS) * q)


def sat_specific_humidity(T_c: float, p_hpa: float) -> float:
    """Saturation specific humidity, g/kg."""
    return q_from_e(sat_vapor_pressure(T_c), p_hpa)


def dewpoint_from_q(q_gkg: float, p_hpa: float) -> float:
    """Dewpoint (C) from specific humidity, by inverting Bolton Eq. 10."""
    e = max(e_from_q(q_gkg, p_hpa), 1e-10)
    ln = math.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln)


def potential_temperature(T_k: float, p_hpa: float) -> float:
    """Potential temperature, K."""
    return T_k * (P0 / p_hpa) ** KAPPA


def temperature_from_theta(theta_k: float, p_hpa: float) -> float:
    """Temperature (K) from potential temperature and pressure."""
    return theta_k * (p_hpa / P0) ** KAPPA


def lcl_pressure(p_hpa: float, T_k: float, q_gkg: float) -> float:
    """
    Pressure (hPa) of the lifting condensation level for a parcel.

    Found by root-finding on "parcel temperature on its dry adiabat equals its
    dewpoint", which is the definition, rather than by an empirical fit. The
    parcel conserves theta and q while unsaturated, so we look for the p where
    saturation specific humidity has fallen to the parcel's actual q.
    """
    theta = potential_temperature(T_k, p_hpa)

    def f(p):
        T_c = temperature_from_theta(theta, p) - 273.15
        return sat_specific_humidity(T_c, p) - q_gkg

    # At the parcel's own level f >= 0 (unsaturated); it decreases upward.
    if f(p_hpa) <= 0:
        return p_hpa  # already saturated
    lo = 1.0
    if f(lo) > 0:
        raise ValueError("LCL not found below 1 hPa; parcel is absurdly dry")
    return brentq(f, lo, p_hpa, xtol=1e-6, rtol=1e-10)


def theta_e_bolton(T_k: float, p_hpa: float, q_gkg: float) -> float:
    """
    Equivalent potential temperature, Bolton (1980) Eq. 43, accurate to ~0.3 K.

    Used to label points on the SP curve and to verify that the pseudoadiabat
    integration conserves what it should. The crude theta*exp(Lv*q/(cp*T))
    approximation drifts by several K over a deep lift and is not good enough
    to test an adiabat with.
    """
    # Bolton's Eq. 43 takes MIXING RATIO, not specific humidity. They differ by
    # q/(1-q), which is 2.8% at 27 g/kg -- small, but it sits inside an
    # exponential, and it showed up as a +1.4 K theta_e drift that looked like
    # an integration error rather than a units slip.
    q = q_gkg / 1000.0
    r = 1000.0 * q / (1.0 - q)  # mixing ratio, g/kg
    e = e_from_q(q_gkg, p_hpa)
    Td = dewpoint_from_q(q_gkg, p_hpa) + 273.15
    T_L = 1.0 / (1.0 / (Td - 56.0) + math.log(T_k / Td) / 800.0) + 56.0
    th_DL = T_k * (1000.0 / (p_hpa - e)) ** 0.2854 * (T_k / T_L) ** (0.28e-3 * r)
    return th_DL * math.exp((3036.0 / T_L - 1.78) * r * 1e-3 * (1 + 0.448e-3 * r))


def moist_lapse_dT_dp(T_k: float, p_hpa: float) -> float:
    """
    Pseudoadiabatic lapse rate expressed as dT/dp (K/hPa).

    Standard saturated-adiabatic form (e.g. Bakhshaii & Stull 2013; AMS
    Glossary), using mixing ratio.
    """
    T = T_k
    es = sat_vapor_pressure(T - 273.15)
    rs = EPS * es / max(p_hpa - es, 1e-6)  # saturation mixing ratio, kg/kg
    Lv = 2.501e6 - 2370.0 * (T - 273.15)   # latent heat, J/kg, temperature-dependent

    # NOTE the denominator's second term is Lv^2 * rs * eps / (Rd * T^2).
    # Writing it with Rv instead of Rd while keeping eps makes it eps (0.622)
    # times too small, which steepens the lapse rate to ~5.3 K/km at 300 K
    # instead of ~3.9 and bleeds ~27 K of theta_e over a deep lift. Parcels
    # then come out under-buoyant, z_fc too high, and PFT too high -- all
    # without anything looking obviously wrong. Rv = Rd/eps, so the two
    # correct spellings are Lv^2*rs/(Rv*T^2) or Lv^2*rs*eps/(Rd*T^2).
    num = (RD * T + Lv * rs)
    den = (CPD + (Lv * Lv * rs * EPS) / (RD * T * T))
    # dT/dp = (1/p) * num/den
    return num / (den * p_hpa)


def moist_adiabat_profile(p_start: float, T_start_k: float,
                          p_levels: np.ndarray) -> np.ndarray:
    """
    Integrate a parcel up a pseudoadiabat from (p_start, T_start_k) and return
    its temperature (K) at each pressure in p_levels.

    p_levels must be sorted DECREASING (upward) and all <= p_start.
    RK4 with a bounded sub-step keeps truncation error well below the ~0.1 K
    that matters here.
    """
    out = np.empty(len(p_levels), dtype=float)
    T = float(T_start_k)
    p = float(p_start)
    max_dp = 2.0  # hPa

    for i, p_target in enumerate(p_levels):
        if p_target > p:
            raise ValueError("moist_adiabat_profile: p_levels must ascend from p_start")
        n = max(1, int(math.ceil((p - p_target) / max_dp)))
        dp = (p_target - p) / n
        for _ in range(n):
            k1 = moist_lapse_dT_dp(T, p)
            k2 = moist_lapse_dT_dp(T + 0.5 * dp * k1, p + 0.5 * dp)
            k3 = moist_lapse_dT_dp(T + 0.5 * dp * k2, p + 0.5 * dp)
            k4 = moist_lapse_dT_dp(T + dp * k3, p + dp)
            T = T + (dp / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            p = p + dp
        p = p_target  # kill accumulated float drift
        out[i] = T
    return out


# ---------------------------------------------------------------------------
# Profile container
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    """
    A single atmospheric column, ordered surface-first (pressure decreasing).

    p_hpa : pressure, hPa
    z_m   : geopotential height above MEAN SEA LEVEL, m
    T_k   : temperature, K
    q_gkg : specific humidity, g/kg
    u_ms, v_ms : wind components, m/s
    """
    p_hpa: np.ndarray
    z_m: np.ndarray
    T_k: np.ndarray
    q_gkg: np.ndarray
    u_ms: np.ndarray
    v_ms: np.ndarray

    def __post_init__(self):
        arrs = [np.asarray(a, dtype=float) for a in
                (self.p_hpa, self.z_m, self.T_k, self.q_gkg, self.u_ms, self.v_ms)]
        n = len(arrs[0])
        if any(len(a) != n for a in arrs):
            raise ValueError("Profile arrays must all be the same length")
        if n < 5:
            raise ValueError("Profile needs at least 5 levels")
        order = np.argsort(-arrs[0])  # descending pressure = ascending height
        (self.p_hpa, self.z_m, self.T_k,
         self.q_gkg, self.u_ms, self.v_ms) = (a[order] for a in arrs)
        if not np.all(np.diff(self.z_m) > 0):
            raise ValueError("Profile heights must be strictly increasing with height")

    def interp_at_p(self, p: float, field: np.ndarray) -> float:
        """Interpolate a field to a pressure, linear in log(p)."""
        lp = np.log(self.p_hpa)
        return float(np.interp(math.log(p), lp[::-1], field[::-1]))

    def T_at_p(self, p: float) -> float:
        return self.interp_at_p(p, self.T_k)

    def z_at_p(self, p: float) -> float:
        return self.interp_at_p(p, self.z_m)

    def p_at_z(self, z: float) -> float:
        return float(np.exp(np.interp(z, self.z_m, np.log(self.p_hpa))))


# ---------------------------------------------------------------------------
# Step 1 -- mixed layer
# ---------------------------------------------------------------------------
def mixed_layer(prof: Profile, max_depth_m: float = 6000.0,
                theta_tol: float = 1.5):
    """
    Iterate the mixed-layer depth until the ML-LCL sits inside the ML.

    Returns (theta_ML [K], q_ML [g/kg], p_mllcl [hPa], z_mllcl [m MSL],
             depth [m], converged [bool], n_levels [int]).

    Averages are weighted LINEARLY IN HEIGHT above the surface, per paper
    Eqs. (15)-(16): entrained mass flux grows linearly with plume height, so
    upper levels contribute more to what the plume actually ingests. The paper
    describes this as a "slight bias toward more elevated levels".

    converged=False means the ML-LCL sits ABOVE the diagnosed mixed layer. With
    the theta cap in place this is the NORMAL case, not an anomaly: dry western
    air routinely has its condensation level well above the mixed layer, and
    all four validation soundings report it. It is a physical statement, not a
    quality flag, and must not be displayed as a warning. The useful quality
    signals are n_levels (an average over 1-2 levels is thin) and the
    caller-computed theta spread.
    """
    z_sfc = prof.z_m[0]
    theta_env = np.array([potential_temperature(t, p)
                          for t, p in zip(prof.T_k, prof.p_hpa)])

    # Cap the search at the top of the ACTUAL mixed layer.
    #
    # The paper's step 1 grows the ML estimate until it contains its own LCL,
    # which is what the loop below does. That works on the paper's own
    # soundings, where a genuine deep mixed layer exists. On real HRRR
    # profiles it runs away: measured over 60 western fires, the median
    # potential-temperature spread across the diagnosed "mixed layer" was
    # 15.7 K and 88% exceeded 3 K. That is not a mixed layer, it is the lower
    # troposphere. Averaging through it dried q_ML to ~1.3-1.8 g/kg, lifting
    # the LCL and hence z_fc, which is SQUARED: median PFT came out 567 GW for
    # those fires against 9.9 GW for the ones whose layer really was mixed.
    #
    # So enforce the definition the method assumes. A mixed layer holds theta
    # near constant; the cap is the first level where theta exceeds the
    # coldest value seen so far by theta_tol. Tracking the running minimum
    # rather than the surface value keeps a superadiabatic surface layer --
    # normal on a hot afternoon -- from ending the layer immediately.
    theta_floor = theta_env[0]
    z_cap = prof.z_m[-1]
    for i in range(1, len(prof.z_m)):
        theta_floor = min(theta_floor, theta_env[i])
        if theta_env[i] > theta_floor + theta_tol:
            z_cap = prof.z_m[i]
            break
    cap_depth = max(z_cap - z_sfc, 200.0)
    max_depth_m = min(max_depth_m, cap_depth)

    best = None
    # Deepen the candidate ML until its own LCL falls within it.
    for depth in np.arange(200.0, max_depth_m + 1.0, 50.0):
        z_top = z_sfc + depth
        if z_top > prof.z_m[-1]:
            break
        mask = prof.z_m <= z_top
        zz = prof.z_m[mask]
        if len(zz) < 2:
            # Too thin to contain a level; fall back to surface values.
            th = theta_env[0]
            qq = prof.q_gkg[0]
        else:
            # Weight linearly in height above the surface. The +1 m floor keeps
            # the surface level from getting exactly zero weight.
            w = (zz - z_sfc) + 1.0
            th = float(np.average(theta_env[mask], weights=w))
            qq = float(np.average(prof.q_gkg[mask], weights=w))

        p_top = prof.p_at_z(z_top)
        T_top = temperature_from_theta(th, p_top)
        try:
            p_lcl = lcl_pressure(p_top, T_top, qq)
        except ValueError:
            continue
        z_lcl = prof.z_at_p(p_lcl)

        best = (th, qq, p_lcl, z_lcl, depth, True, int(mask.sum()))
        if z_lcl <= z_top:
            return best  # ML-LCL now lies inside the ML: converged

    if best is None:
        raise ValueError("mixed_layer: could not form a mixed layer from this profile")
    # Deepest depth tried, ML-LCL still above the ML top. Flag it.
    return (best[0], best[1], best[2], best[3], best[4], False, best[6])


# ---------------------------------------------------------------------------
# Step 2 -- saturation-point curve
# ---------------------------------------------------------------------------
def saturation_point(theta_ML: float, q_ML: float, b: float, prof: Profile):
    """
    The saturation point of plume parcel b on the SP curve (Eqs. 29-30).

    Returns (p_sp [hPa], T_sp [K], z_sp [m MSL], theta_sp [K], q_sp [g/kg]).
    """
    theta_sp = (b + 1.0) * theta_ML
    q_sp = q_ML + b * PHI_GKG * theta_ML

    # The SP is where this parcel saturates. Find p such that qs(T(p), p) = q_sp.
    def f(p):
        T_c = temperature_from_theta(theta_sp, p) - 273.15
        return sat_specific_humidity(T_c, p) - q_sp

    p_hi = prof.p_hpa[0]
    if f(p_hi) <= 0:
        p_sp = p_hi
    else:
        p_sp = brentq(f, 1.0, p_hi, xtol=1e-6, rtol=1e-10)

    T_sp = temperature_from_theta(theta_sp, p_sp)
    return p_sp, T_sp, prof.z_at_p(p_sp), theta_sp, q_sp


# ---------------------------------------------------------------------------
# Step 3 -- free-convection height
# ---------------------------------------------------------------------------
def _cloud_top_pressure(prof: Profile, T_c: float = CLOUD_TOP_C) -> float:
    """
    Pressure of the environmental T = -20 C level (the minimum cloud top).

    Takes the HIGHEST such level (lowest pressure) so that an inversion or a
    shallow cold layer lower down cannot terminate the buoyancy check early.
    """
    target = T_c + 273.15
    cross = []
    for i in range(len(prof.T_k) - 1):
        a, b_ = prof.T_k[i] - target, prof.T_k[i + 1] - target
        if a == 0.0:
            cross.append(prof.p_hpa[i])
        if a * b_ < 0:
            f = a / (a - b_)
            cross.append(prof.p_hpa[i] + f * (prof.p_hpa[i + 1] - prof.p_hpa[i]))
    if not cross:
        raise ValueError(f"Profile never reaches {T_c} C; cannot locate cloud top")
    return min(cross)


def _parcel_clears_environment(prof: Profile, p_sp: float, T_sp: float,
                               p_top: float, n_steps: int = 60) -> float:
    """
    Lift a saturated parcel from its SP to the cloud-top level and return the
    MINIMUM parcel-minus-environment temperature difference (K) along the way.

    >= 0 means the parcel stayed warmer than the environment the whole way.
    """
    if p_sp <= p_top:
        # SP already at or above the -20 C level: nothing to penetrate.
        return float("inf")
    p_levels = np.linspace(p_sp, p_top, n_steps)[1:]
    T_parcel = moist_adiabat_profile(p_sp, T_sp, p_levels)
    T_env = np.array([prof.T_at_p(p) for p in p_levels])
    return float(np.min(T_parcel - T_env))


def free_convection_height(prof: Profile, theta_ML: float, q_ML: float,
                           fire_elev_m: float,
                           dtheta_b: float = DTHETA_B,
                           b_max: float = B_MAX):
    """
    Step 3 + 4. Walk up the SP curve for the lowest parcel that stays buoyant
    from its own SP to the -20 C level, add the buoyancy buffer, and return
    the resulting free-convection height and potential-temperature excess.

    Returns dict with b_fc, z_fc_m (ABOVE FIRE GROUND), dtheta_fc_k, and
    diagnostics.
    """
    p_top = _cloud_top_pressure(prof)

    def margin(b: float) -> float:
        p_sp, T_sp, _, _, _ = saturation_point(theta_ML, q_ML, b, prof)
        return _parcel_clears_environment(prof, p_sp, T_sp, p_top)

    # Buoyancy margin increases monotonically with b (hotter, moister parcel),
    # so a bracketed root find is safe. Scan coarsely first to bracket.
    bs = np.linspace(0.0, b_max, 49)
    margins = np.array([margin(b) for b in bs])

    if margins[0] >= 0.0:
        b_min = 0.0  # even an unmodified ML parcel gets there
    else:
        idx = np.argmax(margins >= 0.0)
        if not np.any(margins >= 0.0):
            raise ValueError(
                "No parcel on the SP curve up to b=%.3f reaches the -20 C level "
                "while staying buoyant. The atmosphere is too stable for this "
                "method to return a finite PFT." % b_max)
        b_min = brentq(margin, bs[idx - 1], bs[idx], xtol=1e-6)

    # The buoyancy buffer: require the parcel to clear the environment by
    # dtheta_b rather than merely to break even. This stands in for the
    # buoyancy the plume loses entraining cooler, drier air.
    def margin_buffered(b: float) -> float:
        return margin(b) - dtheta_b

    if margin_buffered(b_max) < 0.0:
        raise ValueError(
            "No parcel up to b=%.3f clears the environment by the %.2f K buoyancy "
            "buffer." % (b_max, dtheta_b))
    if margin_buffered(b_min) >= 0.0:
        b_fc = b_min
    else:
        b_fc = brentq(margin_buffered, b_min, b_max, xtol=1e-6)

    p_sp, T_sp, z_sp_msl, theta_sp, q_sp = saturation_point(theta_ML, q_ML, b_fc, prof)

    # z_fc is measured ABOVE THE FIRE GROUND. This is the step that a
    # sea-level or sounding-base assumption silently ruins, and z_fc is squared.
    z_fc_m = z_sp_msl - fire_elev_m
    if z_fc_m <= 0:
        raise ValueError(
            "Free-convection height (%.0f m MSL) is at or below the fire ground "
            "(%.0f m). Check fire_elev_m." % (z_sp_msl, fire_elev_m))

    if b_fc <= 1e-6:
        # b_fc = 0 means the UNMODIFIED mixed-layer parcel already stays
        # buoyant to the -20 C level with no fire heat at all. Eq. 31 then
        # multiplies by dtheta_fc = 0 and returns exactly 0 GW, which would
        # display as "no firepower required -- maximally conducive". It is not:
        # it means the column is already deeply convective and the PFT concept
        # does not apply. Refuse rather than emit a degenerate zero.
        raise ValueError(
            "Degenerate: the unmodified mixed-layer parcel is already buoyant "
            "to the -20 C level (b_fc=0), so PFT is identically zero. The "
            "column supports deep convection without a fire; PFT does not "
            "apply here.")

    return {
        "b_fc": b_fc,
        "z_fc_m": z_fc_m,
        "z_fc_msl_m": z_sp_msl,
        "dtheta_fc_k": b_fc * theta_ML,
        "p_fc_hpa": p_sp,
        "T_fc_k": T_sp,
        "q_fc_gkg": q_sp,
        "p_cloudtop_hpa": p_top,
        "b_min_unbuffered": b_min,
    }


# ---------------------------------------------------------------------------
# Step 5 -- mixed-layer wind
# ---------------------------------------------------------------------------
def mean_wind(prof: Profile, z_top_msl: float) -> float:
    """
    Magnitude of the VECTOR-mean wind from the surface to z_top_msl.

    Averaging u and v separately and then taking the magnitude is the point:
    a veering profile must be allowed to cancel. Averaging wind SPEED instead
    inflates U_ML, and U_ML enters PFT linearly.
    """
    mask = prof.z_m <= z_top_msl
    if mask.sum() < 2:
        mask = np.zeros(len(prof.z_m), dtype=bool)
        mask[:2] = True
    zz = prof.z_m[mask]
    uu, vv = prof.u_ms[mask], prof.v_ms[mask]
    # Trapezoidal mean in height: levels are unevenly spaced.
    if len(zz) >= 2 and zz[-1] > zz[0]:
        u_bar = float(np.trapezoid(uu, zz) / (zz[-1] - zz[0]))
        v_bar = float(np.trapezoid(vv, zz) / (zz[-1] - zz[0]))
    else:
        u_bar, v_bar = float(uu.mean()), float(vv.mean())
    return math.hypot(u_bar, v_bar)


# ---------------------------------------------------------------------------
# Step 6 -- PFT
# ---------------------------------------------------------------------------
def pft_from_terms(z_fc_km: float, U_ML: float, dtheta_fc: float) -> float:
    """
    Paper Eq. (31):  PFT(GW) = 0.3 * z_fc(km)^2 * U_ML(m/s) * dtheta_fc(K)

    This is a total firepower, not a firepower per unit area.
    """
    return PFT_COEFF * (z_fc_km ** 2) * U_ML * dtheta_fc


@dataclass
class PFTResult:
    pft_gw: float
    z_fc_km: float
    dtheta_fc_k: float
    u_ml_ms: float
    theta_ml_k: float
    q_ml_gkg: float
    ml_depth_m: float
    ml_lcl_m_agl: float
    b_fc: float
    p_fc_hpa: float
    fire_elev_m: float
    ml_converged: bool
    ml_levels: int

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def compute_pft(prof: Profile, fire_elev_m: float | None = None,
                dtheta_b: float = DTHETA_B) -> PFTResult:
    """
    Full six-step simple-PFT calculation for one column.

    fire_elev_m : elevation of the FIRE GROUND in metres above sea level. If
        omitted, the profile's own lowest level is used -- correct only when
        the column is taken at the fire. Since z_fc is squared, supplying a
        real terrain elevation matters: a 1 km error moves PFT by 2-4x.
    """
    if fire_elev_m is None:
        fire_elev_m = float(prof.z_m[0])

    theta_ML, q_ML, p_mllcl, z_mllcl, depth, converged, n_ml = mixed_layer(prof)
    fc = free_convection_height(prof, theta_ML, q_ML, fire_elev_m, dtheta_b=dtheta_b)
    U_ML = mean_wind(prof, fc["z_fc_msl_m"])
    z_fc_km = fc["z_fc_m"] / 1000.0
    pft = pft_from_terms(z_fc_km, U_ML, fc["dtheta_fc_k"])

    return PFTResult(
        pft_gw=pft,
        z_fc_km=z_fc_km,
        dtheta_fc_k=fc["dtheta_fc_k"],
        u_ml_ms=U_ML,
        theta_ml_k=theta_ML,
        q_ml_gkg=q_ML,
        ml_depth_m=depth,
        ml_lcl_m_agl=z_mllcl - prof.z_m[0],
        b_fc=fc["b_fc"],
        p_fc_hpa=fc["p_fc_hpa"],
        fire_elev_m=fire_elev_m,
        ml_converged=converged,
        ml_levels=n_ml,
    )


# ---------------------------------------------------------------------------
# Published anchor values (paper sections 5a-5c, Figs. 5-7)
#
# These are the authors' own manual analyses. They serve two purposes: they
# unit-test Eq. 31, and they give real events to anchor any display bands to,
# instead of invented percentile cuts.
# ---------------------------------------------------------------------------
BENCHMARKS = [
    # label,                            z_fc_km, dth_fc, U_ML, published_GW
    ("Black Saturday 1000 LST",             4.8,    9.0, 20.0, 1240),
    ("Black Saturday 2200 LST",             3.5,    8.0, 17.0,  500),
    ("Black Saturday 2200 extrapolated",    4.0,    3.0, 20.0,  290),
    ("Chisholm 0600 LST",                   3.3,    8.0, 20.0,  520),
    ("Chisholm 1800 LST",                   2.7,    2.5, 18.0,  100),
    ("Bald Fire",                           4.4,    8.0,  3.0,  140),
    ("Sedgerly Rd Fire",                    2.8,    1.0,  5.0,   12),
]

# Sir Ivan (Feb 2017), an intense pyroCb under extreme fire danger, computed at
# ~300 GW. The authors call this "close to an upper limit of firepower for most
# wildfires", excluding exceptionally large fires. It is the single most useful
# reference point for interpreting a value.
SIR_IVAN_GW = 300.0


def _published_granularity(value: float) -> float:
    """
    The rounding step implied by how a published number is written.

    The paper quotes PFT to one or two significant figures (1240, 500, 12), and
    its inputs to comparable precision. So the honest test is not "within x%"
    but "does our value round to theirs at the precision they published".
    1240 -> 10, 500 -> 100, 12 -> 1.
    """
    v = abs(value)
    if v == 0:
        return 1.0
    step = 1.0
    while v % (step * 10.0) == 0.0:
        step *= 10.0
    return step


def check_benchmarks(verbose: bool = True) -> bool:
    """
    Reproduce the paper's published PFT values from its published terms.

    This tests Eq. 31 and the collapsed constant only -- it does not exercise
    the sounding-based solver. See validate_soundings.py for that.
    """
    ok = True
    if verbose:
        print(f"{'case':36s} {'z_fc':>5s} {'dth':>5s} {'U':>5s} "
              f"{'paper':>7s} {'ours':>8s} {'rounded':>8s}")
        print("-" * 82)
    for label, z, dth, u, published in BENCHMARKS:
        got = pft_from_terms(z, u, dth)
        step = _published_granularity(published)
        rounded = round(got / step) * step
        good = rounded == published
        ok &= good
        if verbose:
            print(f"{label:36s} {z:5.1f} {dth:5.1f} {u:5.1f} "
                  f"{published:7.0f} {got:8.1f} {rounded:8.0f} "
                  f"{'OK' if good else 'FAIL'}")
    if verbose:
        print("-" * 82)
        print(f"Eq.31 coefficient = {PFT_COEFF:.6f} (paper rounds to 0.3)")
        print(f"bracket term      = {PFT_BRACKET:.1f} J/kg/K (paper Table 2: 397.3)")
        print(f"all seven cases reproduce at the paper's published precision: {ok}")
    return bool(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if check_benchmarks() else 1)
