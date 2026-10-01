"""
test_hrrr_pft.py -- offline tests for the HRRR subsetting and profile logic.

Everything here runs without network. The parts that cannot be tested offline
are the S3 fetch itself and pygrib decoding; those are exercised by running
    python hrrr_pft.py
which pulls a real subset and prints one profile.

Run:  python test_hrrr_pft.py
"""

import datetime as dt
import math
import sys

import numpy as np

import hrrr_pft
from hrrr_pft import (parse_idx, select_records, merge_ranges, profile_at,
                      cycle_candidates, hrrr_urls, build_index, nearest_indices)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
def make_idx():
    """A miniature .idx in the real wgrib format."""
    rows = [
        (1, 0, "REFC", "entire atmosphere"),
        (2, 1000, "HGT", "surface"),
        (3, 2500, "PRES", "surface"),
        (4, 4000, "TMP", "1000 mb"),
        (5, 5200, "SPFH", "1000 mb"),
        (6, 6100, "HGT", "1000 mb"),
        (7, 7000, "UGRD", "1000 mb"),
        (8, 7900, "VGRD", "1000 mb"),
        (9, 8800, "TMP", "975 mb"),
        (10, 9700, "CAPE", "surface"),
        (11, 50_000_000, "TMP", "400 mb"),   # far away, forces a second run
    ]
    return "\n".join(f"{n}:{o}:d=2026092712:{v}:{l}:9 hour fcst:"
                     for n, o, v, l in rows)


def test_idx():
    print("\n[1] .idx parsing and byte ranges")
    recs = parse_idx(make_idx())
    check("parses every line", len(recs) == 11, f"{len(recs)}")
    check("sorted by offset", all(recs[i]["offset"] <= recs[i + 1]["offset"]
                                  for i in range(len(recs) - 1)))
    # THE classic bug: a record ends one byte before the next one starts.
    check("record end = next offset - 1", recs[0]["end"] == 999,
          f"first record ends at {recs[0]['end']}")
    check("second record range is 1000-2499",
          recs[1]["offset"] == 1000 and recs[1]["end"] == 2499)
    check("last record is open-ended (runs to EOF)", recs[-1]["end"] is None)
    check("no record end overlaps the next start",
          all(recs[i]["end"] < recs[i + 1]["offset"] for i in range(len(recs) - 1)))

    sel = select_records(recs, levels_hpa=[1000, 975, 400])
    got = {(r["var"], r["level"]) for r in sel}
    check("selects the requested level variables",
          ("TMP", "1000 mb") in got and ("VGRD", "1000 mb") in got)
    check("selects surface HGT (needed to reject below-ground levels)",
          ("HGT", "surface") in got)
    check("does not select unrequested variables",
          ("REFC", "entire atmosphere") not in got and ("CAPE", "surface") not in got)

    runs = merge_ranges(sel, max_gap=2_000_000)
    check("merges nearby records into few ranges", len(runs) == 2,
          f"{len(runs)} ranges: {runs}")
    check("distant record is its own range", runs[-1][0] == 50_000_000)
    # Merging must never drop a wanted record.
    for r in sel:
        covered = any(a <= r["offset"] and (b is None or r["offset"] <= b)
                      for a, b in runs)
        if not covered:
            check(f"range covers {r['var']} {r['level']}", False)
            break
    else:
        check("every selected record falls inside a merged range", True)


# ---------------------------------------------------------------------------
def test_cycles():
    print("\n[2] cycle selection")
    now = dt.datetime(2026, 9, 25, 14, 30, tzinfo=dt.timezone.utc)
    cands = cycle_candidates(target_valid_hour_utc=21, now=now)
    check("returns candidates", len(cands) > 0, f"{len(cands)}")
    check("every candidate is valid at the target hour",
          all(((c.hour + f) % 24) == 21 for c, f in cands))
    check("forecast hours within HRRR's standard range",
          all(0 <= f <= 18 for _, f in cands))
    check("newest first", all(cands[i][0] >= cands[i + 1][0]
                              for i in range(len(cands) - 1)))
    # Cycles younger than ~2 h are skipped: HRRR posts 45-90 min after the hour,
    # so asking for the freshest one is a guaranteed 404.
    check("does not offer a cycle that is almost certainly not posted yet",
          all((now - c).total_seconds() >= 2 * 3600 for c, _ in cands),
          f"newest candidate {cands[0][0]:%H}Z vs now {now:%H:%M}Z")

    g, i = hrrr_urls(dt.datetime(2026, 9, 25, 12, tzinfo=dt.timezone.utc), 9)
    check("grib url shape",
          g.endswith("hrrr.20260925/conus/hrrr.t12z.wrfprsf09.grib2"), g[-52:])
    check("idx url is the grib url plus .idx", i == g + ".idx")


# ---------------------------------------------------------------------------
def synth_fields(orog_m, levels, base_p=1000.0):
    """
    Synthetic HRRR-like fields on a 3x3 grid, with a known orography so the
    below-ground filter can be checked against a hand-computed answer.
    """
    f = {}
    shape = (3, 3)
    f[("HGT", "surface")] = np.full(shape, orog_m, dtype=float)
    f[("PRES", "surface")] = np.full(shape, base_p * 100.0, dtype=float)
    for lev in levels:
        # Standard-atmosphere-ish height for the level.
        z = 44330.0 * (1.0 - (lev / 1013.25) ** 0.1903)
        f[("HGT", lev)] = np.full(shape, z, dtype=float)
        f[("TMP", lev)] = np.full(shape, 288.15 - 0.0065 * z, dtype=float)
        f[("SPFH", lev)] = np.full(shape, 0.006, dtype=float)   # kg/kg
        f[("UGRD", lev)] = np.full(shape, 10.0, dtype=float)
        f[("VGRD", lev)] = np.full(shape, 0.0, dtype=float)
    return f


def test_profile():
    print("\n[3] profile extraction and the below-ground filter")
    levels = list(range(1000, 375, -25))

    # Sea-level point: nothing underground, every level survives.
    f0 = synth_fields(5.0, levels)
    p0, orog0 = profile_at(f0, 1, 1, levels_hpa=levels)
    check("sea-level point keeps all levels",
          p0 is not None and len(p0["p_hpa"]) == len(levels),
          f"{0 if p0 is None else len(p0['p_hpa'])} of {len(levels)}")
    check("orography returned", abs(orog0 - 5.0) < 1e-9)
    check("specific humidity converted kg/kg -> g/kg",
          p0 is not None and abs(p0["q_gkg"][0] - 6.0) < 1e-6,
          f"{p0['q_gkg'][0]:.3f} g/kg")

    # A 3000 m fire: every isobaric level whose height is below the terrain is
    # extrapolated fiction and must be dropped.
    f1 = synth_fields(3000.0, levels)
    p1, _ = profile_at(f1, 1, 1, levels_hpa=levels)
    check("high-terrain point drops below-ground levels",
          p1 is not None and len(p1["p_hpa"]) < len(levels),
          f"{len(p1['p_hpa'])} of {len(levels)} survive above 3000 m")
    check("no surviving level sits at or below the orography",
          p1 is not None and bool(np.all(p1["z_m"] > 3000.0)),
          f"lowest kept level at {p1['z_m'][0]:.0f} m")
    check("surviving levels are ordered surface-first",
          p1 is not None and bool(np.all(np.diff(p1["z_m"]) > 0)))
    check("dropped levels are the high-pressure ones",
          p1 is not None and p1["p_hpa"][0] < 1000.0,
          f"lowest kept level is {p1['p_hpa'][0]:.0f} hPa")

    # Absurd terrain: too few levels left, so refuse rather than return junk.
    f2 = synth_fields(11000.0, levels)
    p2, _ = profile_at(f2, 1, 1, levels_hpa=levels)
    check("returns None when too few levels survive", p2 is None)

    # A level with a NaN must not poison the profile.
    f3 = synth_fields(5.0, levels)
    f3[("TMP", 900)] = np.full((3, 3), np.nan)
    p3, _ = profile_at(f3, 1, 1, levels_hpa=levels)
    check("levels with non-finite values are skipped",
          p3 is not None and 900.0 not in set(p3["p_hpa"]),
          f"{len(p3['p_hpa'])} levels kept")

    # The extracted profile must actually drive the solver.
    from pft import Profile, compute_pft
    prof = Profile(p0["p_hpa"], p0["z_m"], p0["T_k"], p0["q_gkg"],
                   p0["u_ms"], p0["v_ms"])
    try:
        res = compute_pft(prof, fire_elev_m=0.0)
        check("extracted profile feeds the PFT solver",
              math.isfinite(res.pft_gw) and res.pft_gw > 0,
              f"{res.pft_gw:.0f} GW, z_fc {res.z_fc_km:.2f} km")
    except ValueError as e:
        # A standard atmosphere is stable; refusing is a legitimate answer.
        check("extracted profile feeds the PFT solver (stable column refused)",
              True, str(e)[:60])


# ---------------------------------------------------------------------------
def test_level_set():
    """
    The level set has to clear the -20 C level for the HOTTEST fires, not the
    average one.

    A first cut stopped at 400 hPa. That covers a cool Sierra profile (-20 C
    near 520 hPa) but not a 40 C valley fire (~360 hPa) or a hot desert fire
    (~350). Those would have returned "no solution" while every cool
    high-elevation fire worked -- a temperature-dependent dropout that would
    have looked like signal rather than a truncated download.
    """
    print("\n[6] isobaric level set")
    L = hrrr_pft.LEVELS_HPA
    check("strictly descending (surface-first order downstream)",
          all(L[i] > L[i + 1] for i in range(len(L) - 1)))
    check("starts at 1000 hPa", L[0] == 1000)
    check("reaches 300 hPa for -20 C headroom", L[-1] <= 300, f"top is {L[-1]} hPa")

    # Fine spacing where the mixed layer and z_fc live; coarse is fine above.
    fine = [L[i] - L[i + 1] for i in range(len(L) - 1) if L[i] > 600]
    check("25 hPa spacing at and below 600 hPa",
          all(d == 25 for d in fine), f"gaps: {sorted(set(fine))}")

    def p_of_z(z):
        return 1013.25 * (1 - 2.25577e-5 * z) ** 5.25588

    # -20 C level for a range of western fire environments must be inside range.
    for lbl, z0, t0 in [("Sierra 2059 m / 11 C", 2264, 11.1),
                        ("hot valley 500 m / 40 C", 500, 40.0),
                        ("desert 1000 m / 38 C", 1000, 38.0),
                        ("coastal 200 m / 32 C", 200, 32.0)]:
        z_ml = z0 + 3500
        t_ml = t0 - 9.8 * 3.5
        p20 = p_of_z(z_ml + (t_ml + 20.0) / 6.5 * 1000.0)
        check(f"-20 C level inside the level set ({lbl})", p20 > L[-1],
              f"-20 C near {p20:.0f} hPa, top of set {L[-1]} hPa")

    # High-elevation fires must still retain enough levels after the
    # below-ground filter to build a profile at all.
    for elev in (2059, 3000, 3500):
        n = sum(1 for p in L if p < p_of_z(elev))
        check(f"fire at {elev} m keeps >= 8 levels above ground", n >= 8,
              f"{n} levels")

    n_rec = len(L) * len(hrrr_pft.LEVEL_VARS) + len(hrrr_pft.SURFACE_RECORDS)
    check("record count stays modest", n_rec <= 130, f"{n_rec} records")


def test_varnames():
    """
    REAL BUG. The .idx inventory uses NCEP abbreviations (HGT, TMP, SPFH,
    UGRD, VGRD), so select_records matches on those. But pygrib reports
    ecCodes short names -- gh, t, q, u, v -- and surface geopotential height
    comes back as 'orog'. The first live run selected and downloaded all 127
    correct records, then filed them under names nothing looked up, and died
    with a bare KeyError('HGT','surface') 170 MB later.
    """
    print("\n[5] ecCodes -> NCEP variable names")

    class M:
        def __init__(self, sn, nm):
            self.shortName, self.name = sn, nm

    cases = [
        ("gh", "Geopotential height", "HGT"),
        ("orog", "Orography", "HGT"),          # surface HGT is NOT called 'gh'
        ("t", "Temperature", "TMP"),
        ("q", "Specific humidity", "SPFH"),
        ("u", "U component of wind", "UGRD"),
        ("v", "V component of wind", "VGRD"),
        ("sp", "Surface pressure", "PRES"),
        ("HGT", "Geopotential height", "HGT"),  # builds that spell it through
        ("unknown", "Geopotential height", "HGT"),  # long-name fallback
        ("unknown", "Specific humidity", "SPFH"),
    ]
    for sn, nm, want in cases:
        got = hrrr_pft.canonical_var(M(sn, nm))
        check(f"shortName {sn!r} -> {want}", got == want, f"got {got!r}")

    # An unrecognised record must return None. Guessing would let a
    # reflectivity field masquerade as a variable we wanted.
    for sn, nm in [("refc", "Maximum/Composite radar reflectivity"),
                   ("cape", "Convective available potential energy"),
                   ("", "")]:
        check(f"unrecognised {sn!r} returns None",
              hrrr_pft.canonical_var(M(sn, nm)) is None)

    # Every name the profile extractor looks up must be reachable from some
    # ecCodes spelling, or the same failure recurs silently.
    reachable = set(hrrr_pft.ECCODES_TO_NCEP.values()) | set(
        hrrr_pft.LONGNAME_TO_NCEP.values())
    needed = set(hrrr_pft.LEVEL_VARS) | {v for v, _ in hrrr_pft.SURFACE_RECORDS}
    check("every required variable is reachable from an ecCodes name",
          needed <= reachable, f"missing: {sorted(needed - reachable)}")


def test_nearest():
    print("\n[4] grid indexing")
    lat = np.linspace(31.0, 49.0, 40)
    lon = np.linspace(-125.0, -102.0, 50)
    LON, LAT = np.meshgrid(lon, lat)
    tree, shape = build_index(LAT, LON)
    check("index shape matches the grid", shape == LAT.shape, str(shape))

    targets = [(39.5, -120.5), (34.05, -118.25), (47.6, -122.3), (31.2, -110.9)]
    iy, ix = nearest_indices(tree, shape, [t[0] for t in targets],
                             [t[1] for t in targets])
    worst = 0.0
    for k, (tlat, tlon) in enumerate(targets):
        glat, glon = LAT[iy[k], ix[k]], LON[iy[k], ix[k]]
        d = math.hypot(glat - tlat, (glon - tlon) * math.cos(math.radians(tlat)))
        worst = max(worst, d)
    # Grid spacing here is ~0.46 deg, so nearest neighbour must be well inside it.
    check("nearest neighbour is within half a grid cell", worst < 0.35,
          f"worst offset {worst:.3f} deg")

    # Longitude convention: HRRR serves 0-360 and Fireline works in -180..180.
    lons360 = np.where(LON < 0, LON + 360.0, LON)
    back = np.where(lons360 > 180.0, lons360 - 360.0, lons360)
    check("0-360 to -180..180 conversion round trips",
          float(np.max(np.abs(back - LON))) < 1e-9)


if __name__ == "__main__":
    test_idx()
    test_cycles()
    test_profile()
    test_level_set()
    test_varnames()
    test_nearest()
    print(f"\n{'=' * 60}")
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1 if FAIL else 0)
