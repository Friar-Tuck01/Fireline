"""
validate_soundings.py -- end-to-end validation of pft.py against the published
manual analyses in Tory & Kepert (2021).

test_pft.py checks Eq. 31 and the thermodynamics. It does NOT check that the
solver, given a real sounding, finds the same z_fc / dtheta_fc / U_ML that the
authors found by hand on a skew-T. That is what this script is for, and it is
the only test that exercises the whole chain.

Run it from the fireline conda env, which has open network access:

    conda activate fireline
    python validate_soundings.py

Soundings come from the University of Wyoming archive, which is the same source
the paper used (stated in the Fig. 6 caption). If the fetch is blocked, save
each page as text and pass it with --local:

    python validate_soundings.py --local chisholm_12z.txt=CHISHOLM_0600

----------------------------------------------------------------------------
What counts as success
----------------------------------------------------------------------------
The paper's values are MANUAL analyses: the authors read z_fc and dtheta_fc off
a thermodynamic diagram by eye, and estimated U_ML from a visual average of the
wind barbs ("a visual estimate of the average wind speeds in this layer [the
green ellipse] is sufficient"). Their inputs are quoted to 1-2 significant
figures.

So DO NOT expect exact agreement, and do not tune the solver until you get it.
Reproducing the published PFT within a factor of ~1.5, with z_fc within a few
hundred metres and the right ordering between cases, is a real pass. The
authors themselves write:

    "PFT verification has been mainly qualitative. We expect that calculated
     PFT values will contain both biases and random errors... Assuming that
     biases due to inaccurate assumptions are reasonably similar between
     events, useful insight can be gained by considering relative values."

The ORDERING is the strongest signal in this set. Within each event pair, the
morning sounding must give a much larger PFT than the afternoon one:
    Chisholm        520 -> 100 GW   (5.2x drop)
    Black Saturday 1240 -> 500 GW   (2.5x drop)
If the solver reproduces those drops, it is responding to the right physics
even if the absolute values are offset.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

import pft
from pft import Profile, compute_pft

# The University of Wyoming archive retired its old /cgi-bin/sounding endpoint
# ("The legacy interface is no longer available"). The current one is:
#
#   https://weather.uwyo.edu/wsgi/sounding?datetime=YYYY-MM-DD HH:MM:00
#                                         &id=NNNNN&src=SRC&type=TEXT:LIST
#
# No region parameter any more, and the date is one datetime field rather than
# separate YEAR/MONTH/FROM/TO fields. Requesting the old path returns HTTP 404.
UWYO = "https://weather.uwyo.edu/wsgi/sounding"

# src selects the source encoding. UNKNOWN asks the server to autodetect;
# FM35 is the traditional alphanumeric TAC format that the 2001 and 2009
# archives are stored in; BUFR is the modern one. Tried in this order.
SRC_ORDER = ("UNKNOWN", "FM35", "BUFR")

KT_TO_MS = 0.514444

# Set by parse_uwyo() to the archive's own units row, so the wind units can be
# read off the server's output instead of assumed.
_UNITS_ROW = ""

# ---------------------------------------------------------------------------
# The published cases.
#
# fire_elev_m is the crucial one. z_fc is height ABOVE THE FIRE GROUND and it
# is SQUARED, so this number cannot be left to default to the sounding site.
#   - Chisholm: paper states the fire was "about 100 m lower in elevation than
#     the sounding site". Edmonton Stony Plain is at 766 m, so 666 m.
#   - Black Saturday: paper subtracts 0.5 km for the Kinglake and Marysville
#     fires, so 500 m.
# ---------------------------------------------------------------------------
CASES = {
    "CHISHOLM_0600": dict(
        label="Chisholm 0600 LST (12Z 28 May 2001)",
        stn=71119, datetime="2001-05-28 12:00:00",
        fire_elev_m=666.0,
        published=dict(pft_gw=520.0, z_fc_km=3.3, dtheta_fc_k=8.0, u_ml_ms=20.0),
    ),
    "CHISHOLM_1800": dict(
        label="Chisholm 1800 LST (00Z 29 May 2001)",
        stn=71119, datetime="2001-05-29 00:00:00",
        fire_elev_m=666.0,
        published=dict(pft_gw=100.0, z_fc_km=2.7, dtheta_fc_k=2.5, u_ml_ms=18.0),
    ),
    # Melbourne Airport. Two gotchas here:
    #  - The paper cites the Australian BoM number (086282). The Wyoming
    #    archive is keyed on the WMO number, which is 94866.
    #  - The Fig. 5 captions give LAUNCH times (2300 UTC 6 Feb, 1100 UTC
    #    7 Feb). Radiosondes launch ~45-60 min before the nominal synoptic
    #    hour and are ARCHIVED under that hour, so these are the 00Z and 12Z
    #    7 Feb soundings. Melbourne is UTC+11 in February, which is how
    #    1000 LST 7 Feb becomes 2300 UTC 6 Feb.
    "BLACKSAT_1000": dict(
        label="Black Saturday 1000 LST (launched 23Z 6 Feb, archived 00Z 7 Feb 2009)",
        stn=94866, datetime="2009-02-07 00:00:00",
        fire_elev_m=500.0,
        published=dict(pft_gw=1240.0, z_fc_km=4.8, dtheta_fc_k=9.0, u_ml_ms=20.0),
    ),
    "BLACKSAT_2200": dict(
        label="Black Saturday 2200 LST (launched 11Z, archived 12Z 7 Feb 2009)",
        stn=94866, datetime="2009-02-07 12:00:00",
        fire_elev_m=500.0,
        published=dict(pft_gw=500.0, z_fc_km=3.5, dtheta_fc_k=8.0, u_ml_ms=17.0),
    ),
}

# Pairs whose RATIO is the most robust thing to check (see module docstring).
PAIRS = [("CHISHOLM_0600", "CHISHOLM_1800"),
         ("BLACKSAT_1000", "BLACKSAT_2200")]


def build_url(case: dict, src: str) -> str:
    qs = urllib.parse.urlencode({
        "datetime": case["datetime"],
        "id": f"{case['stn']:05d}",
        "src": src,
        "type": "TEXT:LIST",
    })
    return f"{UWYO}?{qs}"


def fetch_sounding(case: dict, timeout: int = 60, dump_prefix: str | None = None):
    """
    Fetch one sounding, trying each source encoding until one parses.

    Returns (text, url, src). Raises RuntimeError with every attempt's outcome
    if none worked, so a failure says which URLs were tried rather than just
    "it didn't work".
    """
    attempts = []
    for src in SRC_ORDER:
        url = build_url(case, src)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "fireline-pft-validation"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            attempts.append(f"src={src}: {type(e).__name__}: {e}")
            continue

        # A 200 is not success -- the archive returns a friendly page when a
        # station or date has no data. Only a parseable table counts.
        try:
            parse_uwyo(text, quiet=True)
            return text, url, src
        except Exception as e:  # noqa: BLE001
            attempts.append(f"src={src}: fetched {len(text)} bytes but "
                            f"{type(e).__name__}: {e}")
            if dump_prefix:
                fn = f"{dump_prefix}_{src}.html"
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(text)
                attempts[-1] += f" [raw response saved to {fn}]"

    raise RuntimeError("no source encoding produced a parseable sounding:\n    "
                       + "\n    ".join(attempts))


def parse_uwyo(text: str, quiet: bool = False) -> Profile:
    """
    Parse the fixed-width table inside the <PRE> block of a UWyo sounding page.

    Columns: PRES HGHT TEMP DWPT RELH MIXR DRCT SKNT THTA THTE THTV
    Rows with missing wind or dewpoint are dropped -- the solver needs all of
    p, z, T, q, u, v at every level it uses, and interpolating across a gap in
    the wind would fabricate shear that was never observed.
    """
    m = re.search(r"<PRE>(.*?)</PRE>", text, re.S | re.I)
    body = m.group(1) if m else text

    # Read the wind units off the archive's own units row instead of assuming
    # them.
    #
    # This is not defensive over-engineering -- it is a bug that already
    # happened. The legacy /cgi-bin interface reported wind speed in KNOTS
    # (column header SKNT). The current /wsgi endpoint reports M/S under the
    # same header. Multiplying the new output by 0.514444 made every wind 49%
    # too small, which propagated straight into PFT because U_ML enters
    # linearly, and it looked like a subtle disagreement with the paper's
    # averaging method rather than a unit error.
    global _UNITS_ROW
    _UNITS_ROW = ""
    for line in body.splitlines():
        low = line.lower()
        if "hpa" in low and ("knot" in low or "m/s" in low or "kt" in low):
            _UNITS_ROW = line.rstrip()
            break

    units_low = _UNITS_ROW.lower()
    if "m/s" in units_low:
        wind_factor = 1.0
    elif "knot" in units_low or "kt" in units_low:
        wind_factor = KT_TO_MS
    else:
        # No units row found. Default to the live endpoint's convention, but
        # say so -- a silent guess here is worth a factor of two in PFT.
        wind_factor = 1.0
        if not quiet:
            print("  WARNING: no units row found; assuming wind is m/s. "
                  "If this page came from the legacy knots interface, PFT will "
                  "be ~1.9x too high.")

    # Two passes, because neither rule alone is enough.
    #
    # Fixed-width 7-char columns first: that is the archive's documented
    # TEXT:LIST layout, and it is the only rule that survives columns running
    # together (a 5-digit height at 100 hPa can touch the pressure field, and
    # whitespace-splitting then sees 10 tokens and silently drops the level).
    #
    # Whitespace-splitting second, as a fallback, in case the column widths
    # change -- the archive already moved servers once and the widths are not
    # a contract.
    #
    # Both rules require exactly 11 numeric fields, which is what makes them
    # safe: a row with a missing value fails the count and is DROPPED, rather
    # than shifting every column left and assigning wind speed to potential
    # temperature.
    def by_fixed_width(line):
        if len(line) < 77:
            return None
        return [line[i:i + 7].strip() for i in range(0, 77, 7)]

    def by_whitespace(line):
        toks = line.split()
        return toks if len(toks) == 11 else None

    rows = []
    for splitter in (by_fixed_width, by_whitespace):
        rows = []
        for line in body.splitlines():
            fields = splitter(line)
            if not fields or len(fields) != 11:
                continue
            try:
                rows.append([float(f) for f in fields])
            except ValueError:
                continue  # header row, units row, or separator
        if len(rows) >= 10:
            break

    if len(rows) < 10:
        raise ValueError(f"Parsed only {len(rows)} usable levels; "
                         "the page may be an error page or the station/date has no data")

    arr = np.array(rows)

    # Real archived soundings contain duplicate pressure levels and occasional
    # non-monotonic heights (the 12Z 28 May 2001 Edmonton sounding does). The
    # Profile class refuses those on purpose, so clean here rather than
    # loosening the invariant: sort by descending pressure, drop duplicate
    # pressures, then keep only levels whose height strictly increases.
    arr = arr[np.argsort(-arr[:, 0])]
    _, keep = np.unique(arr[:, 0], return_index=True)
    arr = arr[np.sort(keep)][::-1] if arr[0, 0] < arr[-1, 0] else arr[np.sort(keep)]
    arr = arr[np.argsort(-arr[:, 0])]

    mono = [0]
    for i in range(1, len(arr)):
        if arr[i, 1] > arr[mono[-1], 1]:
            mono.append(i)
    dropped = len(arr) - len(mono)
    arr = arr[mono]
    if dropped and not quiet:
        print(f"  (dropped {dropped} duplicate/non-monotonic level(s) during parse)")

    p, z, T_c, Td_c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    mixr_gkg, drct, sknt = arr[:, 5], arr[:, 6], arr[:, 7]
    thta = arr[:, 8]

    # Independent check that the columns are what we think they are: the
    # archive publishes its own potential temperature, so recompute it from
    # PRES and TEMP and compare.
    #
    # The comparison must be RELATIVE, not absolute. Potential temperature
    # reaches ~980 K at 6 hPa, where a 0.1 K rounding in TEMP or a fourth-
    # decimal difference in kappa moves theta by ~3 K -- an absolute 1 K
    # threshold rejects perfectly good stratospheric levels and was throwing
    # out three of the four validation soundings. The same kappa difference is
    # worth 0.13 K at 500 hPa, so relative error stays tiny where it matters.
    # A genuine column misalignment is wrong by hundreds of percent, so 2%
    # still catches the failure this check exists for.
    theta_ours = (T_c + 273.15) * (1000.0 / p) ** 0.2854
    rel = float(np.max(np.abs(theta_ours - thta) / np.maximum(thta, 1.0)))
    if rel > 0.02:
        raise ValueError(
            f"Recomputed potential temperature disagrees with the archive's own "
            f"THTA column by up to {rel * 100:.1f}% -- columns are probably "
            f"misaligned, so refusing to build a profile from this page")

    # Mixing ratio -> specific humidity.
    r = mixr_gkg / 1000.0
    q_gkg = 1000.0 * r / (1.0 + r)

    # Wind: meteorological direction (blowing FROM) -> u, v components in m/s,
    # using the conversion selected from the units row above.
    spd = sknt * wind_factor
    rad = np.radians(drct)
    u = -spd * np.sin(rad)
    v = -spd * np.cos(rad)

    _ = Td_c
    return Profile(p, z, T_c + 273.15, q_gkg, u, v)


def report(name: str, case: dict, prof: Profile, src_note: str = "") -> dict:
    pub = case["published"]
    res = compute_pft(prof, fire_elev_m=case["fire_elev_m"])

    print(f"\n{case['label']}")
    if src_note:
        print(f"  source: {src_note}")
    print(f"  sounding: {len(prof.p_hpa)} levels, "
          f"{prof.p_hpa[0]:.0f}-{prof.p_hpa[-1]:.0f} hPa, "
          f"base {prof.z_m[0]:.0f} m MSL; fire ground taken as {case['fire_elev_m']:.0f} m")
    if not res.ml_converged:
        print("  note: ML-LCL lies above the mixed layer (normal in dry air, "
              "not a warning)")
    if getattr(res, "ml_levels", 99) <= 2:
        print(f"  CAUTION: mixed-layer average used only {res.ml_levels} profile "
              f"level(s) -- theta_ML and q_ML are thin")
    print(f"  {'term':<14}{'ours':>10}{'paper':>10}{'ratio':>9}")
    for key, unit in (("z_fc_km", "km"), ("dtheta_fc_k", "K"),
                      ("u_ml_ms", "m/s"), ("pft_gw", "GW")):
        ours, theirs = getattr(res, key), pub[key]
        ratio = ours / theirs if theirs else float("nan")
        print(f"  {key + ' (' + unit + ')':<14}{ours:>10.2f}{theirs:>10.2f}{ratio:>9.2f}")

    # ---- wind diagnostic -------------------------------------------------
    # The published U_ML values run about 2x ours across every case. Step 5 of
    # the paper DEFINES U_ML as a vector mean ("average the meridional and
    # zonal wind components separately and set U_ML to the magnitude of this
    # averaged wind vector"), which is what we compute. But the worked examples
    # ESTIMATED it by eye from the wind barbs -- for Chisholm the text reads
    # "strong south-southeast winds of about 40 kt are evident below z_fc,
    # giving U_ML ~ 20 m/s". Reading a representative barb is much closer to a
    # scalar mean of speed, or to the layer maximum, than to a vector mean.
    #
    # Print all three so the gap is measured, not argued about. If the scalar
    # mean lands near the published value while the vector mean sits at half,
    # the difference is directional shear plus method, not a bug.
    z_top = prof.z_m[0] + res.z_fc_km * 1000.0
    mask = prof.z_m <= z_top
    zz, uu, vv = prof.z_m[mask], prof.u_ms[mask], prof.v_ms[mask]
    spd = np.hypot(uu, vv)
    if len(zz) >= 2 and zz[-1] > zz[0]:
        scalar_mean = float(np.trapezoid(spd, zz) / (zz[-1] - zz[0]))
    else:
        scalar_mean = float(spd.mean())
    u_bar = float(np.trapezoid(uu, zz) / (zz[-1] - zz[0])) if len(zz) >= 2 else float(uu.mean())
    v_bar = float(np.trapezoid(vv, zz) / (zz[-1] - zz[0])) if len(zz) >= 2 else float(vv.mean())
    turn = math.degrees(math.atan2(vv[-1], uu[-1]) - math.atan2(vv[0], uu[0]))
    turn = (turn + 180.0) % 360.0 - 180.0

    print(f"  mixed layer: depth {res.ml_depth_m:.0f} m, ML-LCL {res.ml_lcl_m_agl:.0f} m AGL, "
          f"theta_ML {res.theta_ml_k:.1f} K, q_ML {res.q_ml_gkg:.2f} g/kg, "
          f"converged={res.ml_converged}")
    print(f"  archive units row: {_UNITS_ROW.strip() or '(not found)'}")
    print(f"  wind units taken as: {'m/s' if 'm/s' in _UNITS_ROW.lower() else 'knots'}")
    print(f"  wind in the 0-z_fc layer ({len(zz)} levels):")
    print(f"    vector mean |<u>,<v>|  {math.hypot(u_bar, v_bar):6.2f} m/s   "
          f"<- paper's Step 5 definition, what we use")
    print(f"    scalar mean of speed   {scalar_mean:6.2f} m/s   "
          f"<- closer to reading barbs by eye")
    print(f"    layer max speed        {float(spd.max()):6.2f} m/s")
    print(f"    published U_ML         {pub['u_ml_ms']:6.2f} m/s")
    print(f"    directional turning across layer: {turn:+.0f} deg "
          f"(vector/scalar = {math.hypot(u_bar, v_bar) / scalar_mean:.2f})")

    factor = max(res.pft_gw / pub["pft_gw"], pub["pft_gw"] / res.pft_gw)
    verdict = "within 1.5x" if factor <= 1.5 else (
        "within 2x" if factor <= 2.0 else "OUTSIDE 2x -- investigate")
    print(f"  PFT agreement: {factor:.2f}x  ({verdict})")
    return {"res": res, "published": pub, "factor": factor}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="append", default=[],
                    metavar="FILE=CASE",
                    help="use a saved sounding file instead of fetching, "
                         "e.g. chisholm12z.txt=CHISHOLM_0600")
    ap.add_argument("--case", action="append", default=[],
                    help="run only these cases (default: all)")
    args = ap.parse_args()

    wanted = args.case or list(CASES)
    local_map = {}
    for spec in args.local:
        fn, _, nm = spec.partition("=")
        if nm not in CASES:
            ap.error(f"unknown case {nm!r}; choose from {', '.join(CASES)}")
        local_map[nm] = fn

    print("=" * 72)
    print("PFT end-to-end validation against Tory & Kepert (2021)")
    print("=" * 72)
    print("Paper values are MANUAL skew-T analyses quoted to 1-2 significant")
    print("figures. Agreement within ~1.5x, with the right ordering between")
    print("morning and afternoon, is a pass. Do not tune to match exactly.")

    results = {}
    for name in wanted:
        case = CASES[name]
        try:
            if name in local_map:
                with open(local_map[name], encoding="utf-8", errors="replace") as f:
                    text = f.read()
                src_note = f"local file {local_map[name]}"
            else:
                text, url, src = fetch_sounding(case, dump_prefix=f"raw_{name}")
                src_note = f"src={src}  {url}"
            prof = parse_uwyo(text)
            results[name] = report(name, case, prof, src_note)
        except Exception as e:  # noqa: BLE001 -- report and continue to the next case
            print(f"\n{case['label']}\n  SKIPPED: {type(e).__name__}: {e}")

    # The ratio test. This is the most robust check in the set, because shared
    # biases cancel: the same solver, the same station, hours apart.
    print("\n" + "=" * 72)
    print("Morning/afternoon ratios (shared biases cancel -- the strongest test)")
    print("=" * 72)
    for a, b in PAIRS:
        if a in results and b in results:
            ours = results[a]["res"].pft_gw / results[b]["res"].pft_gw
            theirs = results[a]["published"]["pft_gw"] / results[b]["published"]["pft_gw"]
            ok = "OK" if 0.5 <= ours / theirs <= 2.0 else "CHECK"
            print(f"  {a} / {b}:  ours {ours:5.2f}x   paper {theirs:5.2f}x   {ok}")
        else:
            print(f"  {a} / {b}:  incomplete (a case was skipped)")

    if not results:
        print("\nNo cases ran.")
        print("If a fetch returned bytes that would not parse, the raw response")
        print("was saved as raw_<CASE>_<SRC>.html next to this script -- look at")
        print("one to see what the archive actually returned.")
        print("You can also fetch a sounding by hand from")
        print("  https://weather.uwyo.edu/upperair/sounding.shtml")
        print("save it, and re-run with --local FILE=CASE, e.g.")
        print("  python validate_soundings.py --local chisholm12z.txt=CHISHOLM_0600")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
