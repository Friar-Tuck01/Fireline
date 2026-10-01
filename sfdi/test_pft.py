"""
test_pft.py -- checks for the PFT implementation.

Three layers, in the order that catches bugs earliest:

  1. Thermodynamic primitives against published values and against their own
     definitions (round trips, conservation laws).
  2. The solver's behaviour on synthetic profiles with known answers, built so
     that the right answer can be reasoned out by hand.
  3. Scaling and sensitivity, including the elevation trap that z_fc^2 sets.

Run:  python3 test_pft.py
"""

import math
import sys

import numpy as np

import pft
from pft import (Profile, sat_vapor_pressure, q_from_e, e_from_q,
                 sat_specific_humidity, dewpoint_from_q, potential_temperature,
                 temperature_from_theta, lcl_pressure, moist_adiabat_profile,
                 mixed_layer, saturation_point, free_convection_height,
                 mean_wind, compute_pft, pft_from_terms)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


def close(a, b, tol):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# 1. Thermodynamic primitives
# ---------------------------------------------------------------------------
def test_thermo():
    print("\n[1] Thermodynamic primitives")

    # Saturation vapour pressure against standard table values (Bolton 1980 is
    # accurate to ~0.1% over -35..35 C; the table values below are from the
    # Smithsonian tables as quoted in most texts).
    check("es(0 C) = 6.112 hPa", close(sat_vapor_pressure(0.0), 6.112, 0.001),
          f"{sat_vapor_pressure(0.0):.4f}")
    check("es(20 C) ~ 23.37 hPa", close(sat_vapor_pressure(20.0), 23.37, 0.05),
          f"{sat_vapor_pressure(20.0):.3f}")
    check("es(30 C) ~ 42.43 hPa", close(sat_vapor_pressure(30.0), 42.43, 0.10),
          f"{sat_vapor_pressure(30.0):.3f}")
    check("es(-20 C) ~ 1.25 hPa", close(sat_vapor_pressure(-20.0), 1.254, 0.02),
          f"{sat_vapor_pressure(-20.0):.4f}")
    check("es monotonic in T",
          all(sat_vapor_pressure(t) < sat_vapor_pressure(t + 1)
              for t in range(-40, 40)))

    # q <-> e round trip
    for q0, p0 in [(1.0, 1000.0), (10.0, 850.0), (0.2, 500.0)]:
        e = e_from_q(q0, p0)
        check(f"q<->e round trip (q={q0}, p={p0})",
              close(q_from_e(e, p0), q0, 1e-9))

    # q <-> dewpoint round trip
    for q0, p0 in [(1.0, 1000.0), (12.0, 950.0), (3.0, 700.0)]:
        td = dewpoint_from_q(q0, p0)
        check(f"q<->Td round trip (q={q0}, p={p0})",
              close(sat_specific_humidity(td, p0), q0, 1e-6),
              f"Td={td:.2f} C")

    # Potential temperature: identity at reference pressure, and round trip.
    check("theta(T=300 K, p=1000 hPa) = 300 K",
          close(potential_temperature(300.0, 1000.0), 300.0, 1e-9))
    th = potential_temperature(280.0, 700.0)
    check("theta round trip", close(temperature_from_theta(th, 700.0), 280.0, 1e-9),
          f"theta={th:.2f} K")
    # Textbook: a parcel at 500 hPa with T=-20 C has theta ~ 308 K.
    th500 = potential_temperature(253.15, 500.0)
    check("theta(-20 C, 500 hPa) ~ 308 K", close(th500, 308.5, 1.5), f"{th500:.1f} K")

    # LCL: by definition, at the LCL the parcel is exactly saturated.
    for (p0, T0, q0) in [(1000.0, 300.0, 10.0), (900.0, 305.0, 6.0), (1000.0, 290.0, 8.0)]:
        p_lcl = lcl_pressure(p0, T0, q0)
        theta = potential_temperature(T0, p0)
        T_lcl = temperature_from_theta(theta, p_lcl)
        qs = sat_specific_humidity(T_lcl - 273.15, p_lcl)
        check(f"LCL saturated (p={p0}, T={T0}, q={q0})", close(qs, q0, 1e-5),
              f"p_lcl={p_lcl:.1f} hPa, qs={qs:.5f}")

    # Drier parcel must have a higher (lower-pressure) LCL.
    check("drier parcel -> higher LCL",
          lcl_pressure(1000.0, 300.0, 5.0) < lcl_pressure(1000.0, 300.0, 12.0))

    # Moist adiabat: must be stable-side of the dry adiabat, i.e. a lifted
    # saturated parcel cools MORE SLOWLY than a dry one.
    p_lv = np.array([950.0, 900.0, 850.0, 800.0, 700.0, 600.0, 500.0])
    T_moist = moist_adiabat_profile(1000.0, 300.0, p_lv)
    theta_dry = potential_temperature(300.0, 1000.0)
    T_dry = np.array([temperature_from_theta(theta_dry, p) for p in p_lv])
    check("moist adiabat warmer than dry adiabat aloft", bool(np.all(T_moist > T_dry)),
          f"at 500 hPa: moist {T_moist[-1]:.1f} K vs dry {T_dry[-1]:.1f} K")

    # Warm moist lapse rate near the surface should be ~4-5 K/km, cold/high
    # should approach the dry value of 9.8 K/km.
    dTdp_warm = pft.moist_lapse_dT_dp(300.0, 1000.0)
    # dT/dz = dT/dp * dp/dz, with dp/dz = -rho*g = -p*g/(Rd*T)
    dpdz_warm = -1000.0 * pft.G / (pft.RD * 300.0)
    lapse_warm = -dTdp_warm * dpdz_warm * 1000.0  # K/km
    check("moist lapse at 300 K/1000 hPa is 3.5-5 K/km",
          3.5 < lapse_warm < 5.0, f"{lapse_warm:.2f} K/km")
    dTdp_cold = pft.moist_lapse_dT_dp(233.0, 300.0)
    dpdz_cold = -300.0 * pft.G / (pft.RD * 233.0)
    lapse_cold = -dTdp_cold * dpdz_cold * 1000.0
    check("moist lapse at 233 K/300 hPa is 8-9.8 K/km (near dry)",
          8.0 < lapse_cold < 9.8, f"{lapse_cold:.2f} K/km")

    # The pseudoadiabat must conserve equivalent potential temperature. Use
    # Bolton (1980) Eq. 43, which is accurate to ~0.3 K and is DESIGNED to be
    # conserved along a pseudoadiabat. The crude theta*exp(Lv*q/(cp*T)) form
    # drifts several K over a deep lift all by itself, so testing an adiabat
    # with it says more about the formula than about the integration.
    q_start = sat_specific_humidity(300.0 - 273.15, 1000.0)
    te0 = pft.theta_e_bolton(300.0, 1000.0, q_start)
    for p_end in (700.0, 500.0, 300.0):
        T_end = moist_adiabat_profile(1000.0, 300.0, np.array([p_end]))[0]
        q_end = sat_specific_humidity(T_end - 273.15, p_end)
        te1 = pft.theta_e_bolton(T_end, p_end, q_end)
        check(f"theta_e conserved on pseudoadiabat 1000->{p_end:.0f} hPa",
              close(te0, te1, 0.5), f"{te0:.2f} -> {te1:.2f} K ({te1 - te0:+.2f})")

    # Integration must be resolution-independent at the 0.05 K level.
    a = moist_adiabat_profile(1000.0, 300.0, np.array([500.0]))[0]
    b = moist_adiabat_profile(1000.0, 300.0, np.linspace(1000.0, 500.0, 200)[1:])[-1]
    check("moist adiabat path-independent", close(a, b, 0.05),
          f"direct {a:.4f} vs stepped {b:.4f}")


# ---------------------------------------------------------------------------
# Synthetic profile builder
# ---------------------------------------------------------------------------
def make_profile(sfc_p=1000.0, sfc_T_c=35.0, sfc_z=0.0,
                 zi_m=2500.0, q_ml_gkg=6.0,
                 lapse_k_per_km=6.5, rh_above_pct=25.0,
                 u_ms=10.0, v_ms=0.0, ntop=240,
                 inversion=None):
    """
    Build an idealised AFTERNOON FIRE-WEATHER sounding.

    Structure matters here. An earlier version of this fixture used a constant
    lapse rate and constant RH throughout, which has no mixed layer at all --
    potential temperature increased from the ground up, so the step-1 iteration
    could never converge and always fell through to its max-depth fallback.
    Testing the solver against a profile it is not designed for tests nothing.

    So: a genuinely well-mixed layer (constant theta, constant q) from the
    surface to zi, capped by a stable free atmosphere at a given lapse rate and
    relative humidity. That is what a hot, dry, deeply mixed fire afternoon
    actually looks like on a skew-T.

    inversion : optional (z_base_m_above_sfc, depth_m, dT_K) capping inversion.
    """
    z = np.linspace(sfc_z, sfc_z + 16000.0, ntop)
    zagl = z - sfc_z

    # Mixed layer: constant potential temperature (dry adiabatic in T).
    theta_sfc = (sfc_T_c + 273.15) * (pft.P0 / sfc_p) ** pft.KAPPA
    T = np.empty_like(z)
    dry_lapse = pft.G / pft.CPD * 1000.0  # 9.744 K/km
    in_ml = zagl <= zi_m
    T[in_ml] = (sfc_T_c + 273.15) - dry_lapse * zagl[in_ml] / 1000.0
    T_zi = (sfc_T_c + 273.15) - dry_lapse * zi_m / 1000.0
    T[~in_ml] = T_zi - lapse_k_per_km * (zagl[~in_ml] - zi_m) / 1000.0

    if inversion is not None:
        zb, depth, dT = inversion
        m = (zagl >= zb) & (zagl <= zb + depth)
        T[m] += dT * (zagl[m] - zb) / depth
        T[zagl > zb + depth] += dT

    # Hydrostatic pressure using the layer-mean temperature.
    p = np.empty_like(z)
    p[0] = sfc_p
    for i in range(1, len(z)):
        Tbar = 0.5 * (T[i] + T[i - 1])
        p[i] = p[i - 1] * math.exp(-pft.G * (z[i] - z[i - 1]) / (pft.RD * Tbar))

    # Mixed layer: constant specific humidity. Above: constant RH.
    q = np.empty_like(z)
    q[in_ml] = q_ml_gkg
    q[~in_ml] = [sat_specific_humidity(t - 273.15, pp) * rh_above_pct / 100.0
                 for t, pp in zip(T[~in_ml], p[~in_ml])]
    # Do not let the free atmosphere be moister than the ML top (unphysical jump).
    q[~in_ml] = np.minimum(q[~in_ml], q_ml_gkg)

    _ = theta_sfc  # kept for readability of the construction above
    u = np.full_like(z, u_ms)
    v = np.full_like(z, v_ms)
    return Profile(p, z, T, q, u, v)


# ---------------------------------------------------------------------------
# 2. Solver behaviour
# ---------------------------------------------------------------------------
def test_solver():
    print("\n[2] Solver on synthetic profiles")

    prof = make_profile()
    res = compute_pft(prof)
    check("returns a finite positive PFT",
          math.isfinite(res.pft_gw) and res.pft_gw > 0, f"{res.pft_gw:.1f} GW")
    check("z_fc is physically plausible (1-12 km)",
          1.0 < res.z_fc_km < 12.0, f"{res.z_fc_km:.2f} km")
    check("dtheta_fc is plausible (0-35 K)",
          0.0 < res.dtheta_fc_k < 35.0, f"{res.dtheta_fc_k:.2f} K")
    check("b_fc within the paper's 0-0.1 SP-curve range",
          0.0 <= res.b_fc <= 0.12, f"b={res.b_fc:.4f}")
    check("dtheta_fc equals b_fc * theta_ML (Eq. 29)",
          close(res.dtheta_fc_k, res.b_fc * res.theta_ml_k, 1e-6))

    # Internal consistency: recomputing Eq. 31 from the reported terms must
    # return the reported PFT.
    check("PFT equals Eq.31 applied to its own reported terms",
          close(pft_from_terms(res.z_fc_km, res.u_ml_ms, res.dtheta_fc_k),
                res.pft_gw, 1e-9))

    # The ML-LCL must lie inside the mixed layer -- that is the loop's
    # convergence condition (step 1).
    th, q, p_lcl, z_lcl, depth, converged, n_ml = mixed_layer(prof)

    # THE key invariant, and the one that was missing. A "mixed layer" whose
    # potential temperature spans 16 K is not a mixed layer. Measured over 60
    # real western fires the old code produced a median spread of 15.7 K, which
    # dried q_ML to ~1.3 g/kg, lifted the LCL, lifted z_fc, and inflated PFT by
    # roughly 57x in the median. The cap enforces the definition.
    in_ml = prof.z_m <= prof.z_m[0] + depth
    if in_ml.sum() >= 2:
        th_ml = np.array([potential_temperature(t, p)
                          for t, p in zip(prof.T_k[in_ml], prof.p_hpa[in_ml])])
        spread = float(th_ml.max() - th_ml.min())
    else:
        spread = 0.0
    check("the diagnosed mixed layer is actually mixed",
          spread <= 2.5, f"theta spread {spread:.2f} K across {depth:.0f} m")

    # q_ML must not be dried out by averaging through the free atmosphere.
    # The fixture's mixed layer holds a constant 6 g/kg.
    check("q_ML is not dried by over-deepening", q >= 4.5,
          f"q_ML {q:.2f} g/kg from a 6.0 g/kg mixed layer")
    # How many levels the average actually used. A 1-2 level average is thin,
    # and after the cap that can happen on shallow layers -- so it is reported
    # rather than assumed adequate.
    check("mixed-layer level count is reported", n_ml >= 1, f"{n_ml} levels")
    check("compute_pft exposes ml_levels", isinstance(res.ml_levels, int),
          f"{res.ml_levels} levels")

    # converged now means something honest: the ML-LCL lies inside the ACTUAL
    # mixed layer. A deep, moist ML satisfies it.
    deep_moist = make_profile(q_ml_gkg=6.0, zi_m=4000.0)
    _, _, _, z2, d2, conv2, _ = mixed_layer(deep_moist)
    check("ML-LCL inside the mixed layer reports converged", conv2,
          f"LCL {z2 - deep_moist.z_m[0]:.0f} m AGL, ML depth {d2:.0f} m")
    check("compute_pft reports that convergence",
          compute_pft(deep_moist).ml_converged)

    # And converged=False is now a legitimate physical statement -- ML air will
    # not condense within the mixed layer -- not a failure to be worked around.
    dry_shallow = make_profile(q_ml_gkg=3.0, zi_m=1500.0)
    _, _, _, z3, d3, conv3, _ = mixed_layer(dry_shallow)
    check("LCL above the mixed layer reports not-converged", not conv3,
          f"LCL {z3 - dry_shallow.z_m[0]:.0f} m AGL vs ML depth {d3:.0f} m")
    check("a not-converged column still yields a usable PFT",
          math.isfinite(compute_pft(dry_shallow).pft_gw))

    # A very dry column cannot converge: deepening the ML average dries it
    # further and drives the LCL up faster than the layer grows. Western US
    # fire weather is routinely this dry (2-4 g/kg), so this is a COMMON case,
    # not an exotic one. It must be flagged, not silently absorbed -- the
    # non-converged average biases theta_ML warm and q_ML dry, which biases
    # z_fc and therefore PFT high.
    bone_dry = make_profile(q_ml_gkg=2.0)
    dry_res = compute_pft(bone_dry)
    check("very dry column reports ml_converged=False",
          dry_res.ml_converged is False,
          f"PFT still returned: {dry_res.pft_gw:.0f} GW")
    check("non-converged result is still finite and usable",
          math.isfinite(dry_res.pft_gw) and dry_res.pft_gw > 0)

    # SP curve: hotter parcels must be moister and saturate higher up.
    p_a = saturation_point(th, q, 0.00, prof)[0]
    p_b = saturation_point(th, q, 0.05, prof)[0]
    check("SP curve rises with b (saturates at lower pressure)", p_b < p_a,
          f"b=0: {p_a:.0f} hPa -> b=0.05: {p_b:.0f} hPa")
    q_a = saturation_point(th, q, 0.00, prof)[4]
    q_b = saturation_point(th, q, 0.05, prof)[4]
    check("SP moisture increases with b (Eq. 30)", q_b > q_a,
          f"{q_a:.3f} -> {q_b:.3f} g/kg")
    # Eq. 30 exactly: q_SP - q_ML = b * phi * theta_ML
    check("Eq.30 holds numerically",
          close(q_b - q, 0.05 * pft.PHI_GKG * th, 1e-9))

    # Vector-mean wind must cancel a reversing profile, not average magnitudes.
    p2 = make_profile()
    n = len(p2.u_ms)
    p2.u_ms[: n // 2] = 10.0
    p2.u_ms[n // 2:] = -10.0
    u_rev = mean_wind(p2, p2.z_m[0] + 15000.0)
    check("vector-mean wind cancels a reversing profile", u_rev < 1.5,
          f"{u_rev:.3f} m/s (scalar mean would be 10)")

    # A veering profile: u=10,v=0 below, u=0,v=10 above -> mean (5,5), |.|=7.07
    p3 = make_profile(u_ms=10.0, v_ms=0.0)
    n = len(p3.u_ms)
    p3.u_ms[n // 2:] = 0.0
    p3.v_ms[n // 2:] = 10.0
    u_veer = mean_wind(p3, p3.z_m[0] + 15000.0)
    check("vector-mean wind of a veering profile ~ 7.07 m/s",
          close(u_veer, 7.07, 0.25), f"{u_veer:.3f} m/s")

    # PFT = 0 must never be emitted. b_fc = 0 means the unmodified parcel is
    # already buoyant to -20 C with no fire at all, so Eq. 31 returns exactly
    # zero -- which on a map reads as "maximally conducive" when it actually
    # means the method does not apply. Two real fires in the first live run
    # (Camp Creek, Silvertip) came back 0.0 GW this way.
    for q in (10.0, 12.0, 14.0):
        for zi in (3000.0, 4000.0, 5000.0):
            try:
                r = compute_pft(make_profile(q_ml_gkg=q, zi_m=zi))
                if r.pft_gw <= 0.0:
                    check("degenerate zero PFT is never returned", False,
                          f"q={q} zi={zi} gave {r.pft_gw} GW")
                    break
            except ValueError:
                pass
        else:
            continue
        break
    else:
        check("degenerate zero PFT is never returned", True,
              "refused rather than emitting 0 GW")

    # A profile too stable to reach -20 C must raise, not silently return junk.
    stable = make_profile(lapse_k_per_km=1.0, q_ml_gkg=1.0, sfc_T_c=15.0,
                          zi_m=500.0, rh_above_pct=5.0)
    try:
        compute_pft(stable)
        check("very stable profile raises rather than fabricating a value", False,
              "no exception raised")
    except ValueError as e:
        check("very stable profile raises rather than fabricating a value", True,
              str(e)[:60])


# ---------------------------------------------------------------------------
# 3. Scaling, sensitivity, and the elevation trap
# ---------------------------------------------------------------------------
def test_scaling():
    print("\n[3] Scaling and sensitivity")

    # Eq. 31 scalings, tested directly.
    base = pft_from_terms(3.0, 10.0, 5.0)
    check("PFT scales as z_fc^2", close(pft_from_terms(6.0, 10.0, 5.0), 4 * base, 1e-9))
    check("PFT scales linearly in U_ML",
          close(pft_from_terms(3.0, 20.0, 5.0), 2 * base, 1e-9))
    check("PFT scales linearly in dtheta_fc",
          close(pft_from_terms(3.0, 10.0, 10.0), 2 * base, 1e-9))

    # Stronger wind must RAISE the threshold: a windier day needs a bigger fire
    # to go deep, because the plume is tilted over.
    calm = compute_pft(make_profile(u_ms=3.0))
    windy = compute_pft(make_profile(u_ms=25.0))
    check("stronger wind raises PFT (harder to reach deep convection)",
          windy.pft_gw > calm.pft_gw,
          f"3 m/s -> {calm.pft_gw:.0f} GW, 25 m/s -> {windy.pft_gw:.0f} GW")

    # A moister boundary layer should lower the threshold: the plume condenses
    # sooner, so z_fc drops and PFT falls with its square.
    dry = compute_pft(make_profile(q_ml_gkg=2.0))
    moist = compute_pft(make_profile(q_ml_gkg=8.0))
    check("moister mixed layer lowers PFT",
          moist.pft_gw < dry.pft_gw,
          f"q=2 g/kg -> {dry.pft_gw:.0f} GW, q=8 g/kg -> {moist.pft_gw:.0f} GW")
    check("moister mixed layer lowers z_fc",
          moist.z_fc_km < dry.z_fc_km,
          f"{dry.z_fc_km:.2f} km -> {moist.z_fc_km:.2f} km")

    # A capping inversion must raise the threshold.
    nocap = compute_pft(make_profile())
    capped = compute_pft(make_profile(inversion=(1500.0, 400.0, 4.0)))
    check("capping inversion raises PFT",
          capped.pft_gw > nocap.pft_gw,
          f"{nocap.pft_gw:.0f} -> {capped.pft_gw:.0f} GW")

    # THE ELEVATION TRAP. z_fc is measured above the FIRE, and it is squared.
    # Same atmosphere, fire 1500 m higher up the hill: PFT must drop sharply.
    prof = make_profile(sfc_z=0.0)
    at_sea = compute_pft(prof, fire_elev_m=0.0)
    on_hill = compute_pft(prof, fire_elev_m=1500.0)
    ratio = at_sea.pft_gw / on_hill.pft_gw
    check("fire elevation changes PFT via z_fc^2",
          ratio > 1.5,
          f"0 m -> {at_sea.pft_gw:.0f} GW, 1500 m -> {on_hill.pft_gw:.0f} GW "
          f"({ratio:.2f}x)")
    check("elevation enters only through z_fc, not the other terms",
          close(at_sea.u_ml_ms, on_hill.u_ml_ms, 1e-6)
          and close(at_sea.dtheta_fc_k, on_hill.dtheta_fc_k, 1e-6))
    # And the relationship really is the square of the height difference.
    expected = (at_sea.z_fc_km / on_hill.z_fc_km) ** 2
    check("PFT ratio equals the z_fc ratio squared",
          close(ratio, expected, 1e-6), f"{ratio:.4f} vs {expected:.4f}")

    # The buoyancy buffer should push the threshold up, never down.
    p = make_profile()
    lo = compute_pft(p, dtheta_b=0.5)
    hi = compute_pft(p, dtheta_b=1.0)
    check("larger buoyancy buffer raises PFT",
          hi.pft_gw >= lo.pft_gw,
          f"db=0.5 -> {lo.pft_gw:.0f} GW, db=1.0 -> {hi.pft_gw:.0f} GW")


def test_uwyo_parser():
    """
    Render a known profile in University of Wyoming fixed-width format, parse
    it back, and confirm the round trip.

    This is what catches a column-offset slip, a mixing-ratio/specific-humidity
    mix-up, or an inverted wind convention -- all of which would produce a
    profile that parses without error and gives a confidently wrong PFT.
    """
    print("\n[5] UWyo sounding parser round trip")
    try:
        import validate_soundings as vs
    except ImportError:
        check("validate_soundings importable", False, "module not found")
        return

    # v = -12 m/s with u = 0 is a wind FROM the north, i.e. direction 360.
    prof = make_profile(q_ml_gkg=6.0, u_ms=0.0, v_ms=-12.0)
    step = 6
    lines = ["<PRE>",
             "   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   "
             "THTA   THTE   THTV",
             "   hPa     m      C      C      %    g/kg    deg   knot     "
             "K      K      K "]
    for i in range(0, len(prof.p_hpa), step):
        pr, z = prof.p_hpa[i], prof.z_m[i]
        T_c, q = prof.T_k[i] - 273.15, prof.q_gkg[i]
        td = pft.dewpoint_from_q(q, pr)
        mixr = q / (1000.0 - q) * 1000.0
        spd_kt = math.hypot(prof.u_ms[i], prof.v_ms[i]) / 0.514444
        drct = math.degrees(math.atan2(-prof.u_ms[i], -prof.v_ms[i])) % 360.0
        th = pft.potential_temperature(prof.T_k[i], pr)
        # Match the archive's real widths: HGHT, DRCT and SKNT print as
        # integers. Rendering height as %7.1f gives "16000.0", which fills the
        # field exactly and runs into the pressure column -- that is a bug in
        # the FIXTURE, not the parser, and it is how the two-pass splitter in
        # validate_soundings came to exist.
        lines.append(
            f"{pr:7.1f}{z:7.0f}{T_c:7.1f}{td:7.1f}{50.0:7.0f}{mixr:7.2f}"
            f"{drct:7.0f}{spd_kt:7.0f}{th:7.1f}{th:7.1f}{th:7.1f}")
    lines.append("</PRE>")

    parsed = vs.parse_uwyo("\n".join(lines))
    expect_n = len(range(0, len(prof.p_hpa), step))
    check("parser recovers every level", len(parsed.p_hpa) == expect_n,
          f"{len(parsed.p_hpa)} of {expect_n}")
    # Tolerances are set by the %7.1f rendering above, not by the parser.
    check("pressure round trip",
          float(np.max(np.abs(parsed.p_hpa - prof.p_hpa[::step]))) < 0.06)
    check("temperature round trip",
          float(np.max(np.abs(parsed.T_k - prof.T_k[::step]))) < 0.06)
    check("mixing-ratio -> specific-humidity conversion round trips",
          float(np.max(np.abs(parsed.q_gkg - prof.q_gkg[::step]))) < 0.06)
    # Tolerance is set by INTEGER-KNOT QUANTIZATION in the source, not by the
    # parser. The archive reports SKNT as whole knots, so wind components carry
    # up to 0.5 kt = 0.26 m/s of rounding. U_ML enters PFT linearly, so this is
    # a real ~2% floor on wind-driven precision for a 12 m/s layer -- worth
    # knowing, and not something to tune away.
    du = float(np.max(np.abs(parsed.u_ms - prof.u_ms[::step])))
    dv = float(np.max(np.abs(parsed.v_ms - prof.v_ms[::step])))
    check("wind direction convention (from-direction -> u,v) round trips",
          du < 0.3 and dv < 0.3,
          f"max du={du:.3f}, dv={dv:.3f} m/s (integer-knot rounding is 0.26 m/s)")
    # Direction must be preserved exactly in sign: a north wind is still a
    # north wind, not a south wind. This is the part a sign error would break.
    check("wind sign preserved (north wind is not flipped to south)",
          bool(np.all(parsed.v_ms < 0)) and bool(np.all(np.abs(parsed.u_ms) < 0.3)),
          f"v ranges {parsed.v_ms.min():.2f} to {parsed.v_ms.max():.2f} m/s")

    a = compute_pft(prof, fire_elev_m=0.0)
    b = compute_pft(parsed, fire_elev_m=0.0)
    check("PFT survives the round trip within 5%",
          abs(a.pft_gw - b.pft_gw) / a.pft_gw < 0.05,
          f"{a.pft_gw:.1f} vs {b.pft_gw:.1f} GW (coarser levels + print rounding)")

    # --- wind units regression -------------------------------------------
    # REAL BUG, caught only by validating against archived soundings: the
    # legacy /cgi-bin interface reported wind speed in KNOTS under the header
    # SKNT; the current /wsgi endpoint reports M/S under the same header.
    # Applying the knots conversion to m/s data made every wind 49% too small,
    # and since U_ML enters PFT linearly, every PFT was ~half what it should
    # have been. It looked like a methodological disagreement with the paper
    # for two rounds of analysis before the units row settled it.
    #
    # Build the SAME profile as both a knots page and an m/s page. Both must
    # come back with the same wind.
    def render(units):
        tok = "m/s" if units == "ms" else "knot"
        f = 1.0 if units == "ms" else 1.0 / 0.514444
        out = ["<PRE>",
               "   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   "
               "THTA   THTE   THTV",
               f"    hPa     m      C      C      %    g/kg    deg  {tok:>5s}"
               "     K      K      K "]
        for i in range(0, len(prof.p_hpa), step):
            pr, z = prof.p_hpa[i], prof.z_m[i]
            T_c, q = prof.T_k[i] - 273.15, prof.q_gkg[i]
            td = pft.dewpoint_from_q(q, pr)
            mixr = q / (1000.0 - q) * 1000.0
            sp = math.hypot(prof.u_ms[i], prof.v_ms[i]) * f
            dr = math.degrees(math.atan2(-prof.u_ms[i], -prof.v_ms[i])) % 360.0
            th = pft.potential_temperature(prof.T_k[i], pr)
            out.append(f"{pr:7.1f}{z:7.0f}{T_c:7.1f}{td:7.1f}{50.0:7.0f}"
                       f"{mixr:7.2f}{dr:7.0f}{sp:7.1f}{th:7.1f}{th:7.1f}{th:7.1f}")
        out.append("</PRE>")
        return "\n".join(out)

    p_ms = vs.parse_uwyo(render("ms"), quiet=True)
    p_kt = vs.parse_uwyo(render("kt"), quiet=True)
    truth = prof.v_ms[0]
    check("m/s page parses wind correctly",
          abs(p_ms.v_ms[0] - truth) < 0.05, f"{p_ms.v_ms[0]:.3f} vs {truth:.3f}")
    check("knots page parses wind correctly",
          abs(p_kt.v_ms[0] - truth) < 0.05, f"{p_kt.v_ms[0]:.3f} vs {truth:.3f}")
    check("both unit conventions give the same wind",
          abs(p_ms.v_ms[0] - p_kt.v_ms[0]) < 0.05,
          "units are read from the page, not assumed")
    # Relative, not absolute: the knots rendering rounds to 0.1 kt, which is
    # 0.013 m/s of wind and therefore ~0.1% of a 1300 GW PFT.
    pft_ms = compute_pft(p_ms, fire_elev_m=0.0).pft_gw
    pft_kt = compute_pft(p_kt, fire_elev_m=0.0).pft_gw
    check("both unit conventions give the same PFT",
          abs(pft_ms - pft_kt) / pft_ms < 0.005,
          f"{pft_ms:.1f} vs {pft_kt:.1f} GW")


def test_benchmarks():
    print("\n[4] Published benchmarks (paper Figs. 5-7)")
    ok = pft.check_benchmarks(verbose=False)
    for label, z, dth, u, published in pft.BENCHMARKS:
        got = pft_from_terms(z, u, dth)
        step = pft._published_granularity(published)
        check(f"{label} = {published} GW", round(got / step) * step == published,
              f"computed {got:.1f} GW")
    check("all published cases reproduce", ok)


if __name__ == "__main__":
    test_thermo()
    test_solver()
    test_scaling()
    test_uwyo_parser()
    test_benchmarks()
    print(f"\n{'=' * 60}")
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    sys.exit(1 if FAIL else 0)
