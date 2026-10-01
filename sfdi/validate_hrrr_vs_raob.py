"""
validate_hrrr_vs_raob.py -- does the PRODUCTION path agree with the reference?

Every validation so far used radiosondes. What actually ships reads HRRR model
profiles at 25 hPa spacing, drops below-ground levels, and samples by nearest
neighbour on a 3 km grid. None of that had been tested against a known answer.
This closes that gap.

For each western radiosonde site, at a synoptic hour, it computes PFT twice:

    RAOB : the observed sounding, the same path validate_soundings.py uses
    HRRR : the f00 analysis at the same time, sampled at the station, the same
           path compute_pft_fires.py uses for every fire

Same solver, same elevation, two data sources. Any disagreement is the HRRR
path, which is exactly what needs measuring.

    conda activate fireline
    python validate_hrrr_vs_raob.py --date 2026-09-28 --hour 12
    python validate_hrrr_vs_raob.py --days 3
    python validate_hrrr_vs_raob.py --date 2026-09-28 --hour 12 --sites 72489,72681

----------------------------------------------------------------------------
Design notes
----------------------------------------------------------------------------
* Station coordinates and elevation are PARSED FROM THE SOUNDING PAGE, not
  hardcoded. A table of lat/lons typed from memory is exactly the kind of thing
  that has already gone wrong once in this project (the retired UWyo endpoint),
  and a wrong station location would produce a quiet, plausible disagreement
  rather than an error.

* Both sides use the SAME fire-ground elevation -- the station elevation. That
  isolates the profile difference from the terrain-representation difference.
  HRRR's orography at the point is reported separately, since on a 3 km grid it
  can sit far from the true station height, and z_fc is squared.

* One HRRR download serves every site in a cycle. The subset is ~150 MB, so
  fetching per-site would be absurd.

----------------------------------------------------------------------------
Reading the result
----------------------------------------------------------------------------
The ratio HRRR/RAOB is the number that matters. Perfect agreement is not
expected: a 3 km model analysis is not a balloon, and the balloon drifts
downwind as it climbs. What would be reassuring is a median ratio near 1 with
most sites inside 1.5x and no systematic bias. What would be alarming is a
median far from 1 (the HRRR path biased) or a huge spread (the path unstable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

import pft
from pft import Profile, compute_pft
import hrrr_pft
from validate_soundings import parse_uwyo, UWYO

# The archive's own links use src=BUFR, so try that first.
SRC_ORDER = ("BUFR", "UNKNOWN", "FM35")

# Western US radiosonde sites, by WMO number. Coordinates are deliberately NOT
# listed -- they are read from each sounding page. A site that does not resolve
# is reported and skipped, so a wrong id here costs a line of output, not a
# wrong answer.
WESTERN_RAOB = {
    72797: "UIL Quillayute WA",
    72786: "OTX Spokane WA",
    72694: "SLE Salem OR",
    72597: "MFR Medford OR",
    72493: "OAK Oakland CA",
    72393: "VBG Vandenberg CA",
    72293: "NKX San Diego CA",
    72489: "REV Reno NV",
    72582: "LKN Elko NV",
    72387: "DRA Desert Rock NV",
    72681: "BOI Boise ID",
    72572: "SLC Salt Lake City UT",
    72776: "TFX Great Falls MT",
    72672: "RIW Riverton WY",
    72476: "GJT Grand Junction CO",
    72469: "DNR Denver CO",
    72376: "FGZ Flagstaff AZ",
    72274: "TWC Tucson AZ",
    72365: "ABQ Albuquerque NM",
}

UA = {"User-Agent": "fireline-pft-validation/1.0"}


# ---------------------------------------------------------------------------
def fetch_raob(stn: int, when: dt.datetime, timeout=60, pause=1.2):
    """
    Sounding text for one station and time.

    Two changes from the naive version, both learned the hard way:

    * PAUSE between requests. Nineteen sites times three source encodings is up
      to 57 rapid hits on a public archive, and the first run had eleven sites
      fail with HTTP errors that looked like missing data but were almost
      certainly throttling.
    * Report the actual HTTP STATUS. "HTTPError" alone cannot distinguish "this
      station has no 12Z sounding" (404) from "you are asking too fast" (429 or
      503), and those call for opposite responses.
    """
    errs = []
    for src in SRC_ORDER:
        qs = urllib.parse.urlencode({
            "datetime": when.strftime("%Y-%m-%d %H:00:00"),
            "id": f"{stn:05d}", "src": src, "type": "TEXT:LIST"})
        url = f"{UWYO}?{qs}"
        try:
            time.sleep(pause)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            errs.append(f"{src}:HTTP {e.code}")
            # A 404 means this station/time genuinely has nothing; trying the
            # other encodings just adds load. Throttling deserves a retry.
            if e.code == 404:
                break
            continue
        except Exception as e:  # noqa: BLE001
            errs.append(f"{src}:{type(e).__name__}")
            continue
        try:
            parse_uwyo(text, quiet=True)
            return text
        except Exception as e:  # noqa: BLE001
            errs.append(f"{src}:unparseable({str(e)[:30]})")
    raise RuntimeError("; ".join(errs))


def station_latlon(text: str, save_on_fail: str | None = None):
    """
    Station latitude and longitude from the sounding page.

    The current wsgi page carries them in one italic line:

        <BR/><I>Latitude: 39.568 Longitude: -119.795</I>

    It does NOT carry station elevation. The only "elevation" text in the page
    is a LEGEND TABLE listing what SLAT/SLON/SELV mean, with no values -- and
    an earlier version of this function demanded all three, so it threw away
    eight sites that had perfectly good coordinates. Elevation now comes from
    the sounding's own first level (see station_elevation), which is the launch
    point and is self-consistent with the profile rather than scraped prose.

    Matching latitude and longitude TOGETHER in one pattern is deliberate: it
    anchors on the real data line and cannot be satisfied by the legend table,
    where the words appear without numbers.
    """
    m = re.search(
        r"Latitude\s*:\s*(-?\d+(?:\.\d+)?)\s*(?:</?[A-Za-z/][^>]*>\s*)*"
        r"Longitude\s*:\s*(-?\d+(?:\.\d+)?)", text, re.I)
    if not m:
        # Fall back to the abbreviated form, but only where a NUMBER follows,
        # so the legend table cannot match.
        la = re.search(r"\bSLAT\b[^0-9\-]{0,12}(-?\d+(?:\.\d+)?)", text, re.I)
        lo = re.search(r"\bSLON\b[^0-9\-]{0,12}(-?\d+(?:\.\d+)?)", text, re.I)
        if not (la and lo):
            if save_on_fail:
                with open(save_on_fail, "w", encoding="utf-8") as f:
                    f.write(text)
            raise ValueError("latitude/longitude not found in page"
                             + (f" [saved to {save_on_fail}]" if save_on_fail else ""))
        lat, lon = float(la.group(1)), float(lo.group(1))
    else:
        lat, lon = float(m.group(1)), float(m.group(2))

    # A positive longitude in the western US would silently sample HRRR on the
    # far side of the planet, where the nearest-neighbour lookup returns a grid
    # point rather than an error.
    if lon > 0 and 20 < lat < 75:
        lon = -lon
    return lat, lon


def station_elevation(prof) -> float:
    """
    Station elevation from the sounding itself.

    The lowest reported level of a radiosonde IS the launch point, so this is
    the station height by construction -- no scraping, and guaranteed
    consistent with the profile the solver is about to use.
    """
    return float(prof.z_m[0])


def hrrr_analysis(cycle: dt.datetime, workdir=".", verbose=True):
    """
    Download and read the HRRR f00 analysis subset for one cycle.

    f00 at the sounding's own valid hour, so the two are contemporaneous --
    no forecast error mixed into the comparison.
    """
    grib_url, idx_url = hrrr_pft.hrrr_urls(cycle, 0)
    out_path = os.path.join(workdir, f"hrrr_raob_{cycle:%Y%m%d%H}_f00.grib2")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        if verbose:
            print(f"  reusing cached {out_path} "
                  f"({os.path.getsize(out_path) / 1e6:.0f} MB)")
    else:
        idx = hrrr_pft.parse_idx(
            hrrr_pft._get(idx_url, timeout=45).decode("utf-8", errors="replace"))
        want = hrrr_pft.select_records(idx)
        runs = hrrr_pft.merge_ranges(want)
        if verbose:
            print(f"  downloading {len(want)} records in {len(runs)} range(s) ...",
                  flush=True)
        n = hrrr_pft.download_subset(grib_url, runs, out_path)
        if verbose:
            print(f"  {n / 1e6:.0f} MB -> {out_path}")
    return hrrr_pft.read_fields(out_path)


def pft_from(profile_dict, elev, label):
    """Run the solver, turning refusals into a short status string."""
    try:
        prof = Profile(profile_dict["p_hpa"], profile_dict["z_m"],
                       profile_dict["T_k"], profile_dict["q_gkg"],
                       profile_dict["u_ms"], profile_dict["v_ms"])
        return compute_pft(prof, fire_elev_m=elev), None
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        st = ("degenerate" if msg.startswith("Degenerate")
              else "too_stable" if "never reaches" in msg else "error")
        return None, f"{label}:{st}"


def run_cycle(when: dt.datetime, sites, workdir=".", verbose=True):
    """Compare RAOB and HRRR PFT at every site for one synoptic hour."""
    print(f"\n{'=' * 92}")
    print(f"{when:%Y-%m-%d %H}Z   HRRR f00 analysis vs radiosonde")
    print(f"{'=' * 92}")

    try:
        fields, lats, lons = hrrr_analysis(when, workdir=workdir, verbose=verbose)
    except Exception as e:  # noqa: BLE001
        print(f"  HRRR unavailable for this cycle: {type(e).__name__}: {e}")
        return []
    tree, shape = hrrr_pft.build_index(lats, lons)

    print(f"\n  {'site':22s}{'elev':>6}{'orog':>6}"
          f"{'PFT raob':>10}{'PFT hrrr':>10}{'ratio':>7}"
          f"{'zfc r':>7}{'dth r':>7}{'U r':>6}  note")
    print("  " + "-" * 88)

    rows = []
    for stn, label in sites.items():
        try:
            text = fetch_raob(stn, when)
            raob_prof = parse_uwyo(text, quiet=True)
            lat, lon = station_latlon(
                text, save_on_fail=f"raw_raob_{stn}_{when:%Y%m%d%H}.html")
            elev = station_elevation(raob_prof)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:22s}{'':>6}{'':>6}{'':>10}{'':>10}{'':>7}"
                  f"{'':>7}{'':>7}{'':>6}  RAOB unavailable ({str(e)[:28]})")
            continue

        iy, ix = hrrr_pft.nearest_indices(tree, shape, [lat], [lon])
        hp, orog = hrrr_pft.profile_at(fields, int(iy[0]), int(ix[0]))
        if hp is None:
            print(f"  {label:22s}{elev:6.0f}{orog:6.0f}{'':>10}{'':>10}{'':>7}"
                  f"{'':>7}{'':>7}{'':>6}  too few HRRR levels above ground")
            continue

        # SAME elevation on both sides, so the comparison isolates the profile.
        r_raob, e1 = pft_from({"p_hpa": raob_prof.p_hpa, "z_m": raob_prof.z_m,
                               "T_k": raob_prof.T_k, "q_gkg": raob_prof.q_gkg,
                               "u_ms": raob_prof.u_ms, "v_ms": raob_prof.v_ms},
                              elev, "raob")
        r_hrrr, e2 = pft_from(hp, elev, "hrrr")

        if r_raob is None or r_hrrr is None:
            note = " ".join(x for x in (e1, e2) if x)
            pr = f"{r_raob.pft_gw:10.1f}" if r_raob else f"{'-':>10}"
            ph = f"{r_hrrr.pft_gw:10.1f}" if r_hrrr else f"{'-':>10}"
            print(f"  {label:22s}{elev:6.0f}{orog:6.0f}{pr}{ph}{'':>7}"
                  f"{'':>7}{'':>7}{'':>6}  {note}")
            continue

        ratio = r_hrrr.pft_gw / r_raob.pft_gw if r_raob.pft_gw > 0 else float("nan")
        zr = r_hrrr.z_fc_km / r_raob.z_fc_km if r_raob.z_fc_km else float("nan")
        dr = (r_hrrr.dtheta_fc_k / r_raob.dtheta_fc_k
              if r_raob.dtheta_fc_k else float("nan"))
        ur = r_hrrr.u_ml_ms / r_raob.u_ml_ms if r_raob.u_ml_ms else float("nan")
        flags = []
        if r_hrrr.ml_levels <= 2:
            flags.append("hrrr ML thin")
        if abs(orog - elev) > 200:
            flags.append(f"orog off {orog - elev:+.0f} m")
        print(f"  {label:22s}{elev:6.0f}{orog:6.0f}"
              f"{r_raob.pft_gw:10.1f}{r_hrrr.pft_gw:10.1f}{ratio:7.2f}"
              f"{zr:7.2f}{dr:7.2f}{ur:6.2f}  {', '.join(flags)}")
        rows.append({"site": label, "ratio": ratio, "zr": zr, "dr": dr, "ur": ur,
                     "thin": bool(r_hrrr.ml_levels <= 2),
                     "raob": r_raob.pft_gw, "hrrr": r_hrrr.pft_gw,
                     "elev": elev, "orog": orog})
    return rows


def summarise(rows):
    if not rows:
        print("\nNo comparable pairs. Nothing to conclude.")
        return 1
    r = np.array([x["ratio"] for x in rows], dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    print(f"\n{'=' * 92}")
    print(f"AGGREGATE  ({len(r)} site-hours)")
    print(f"{'=' * 92}")

    # Geometric statistics: a ratio of 2 and a ratio of 0.5 are equally wrong,
    # and only the log treats them that way.
    lg = np.log(r)
    print(f"  median HRRR/RAOB ratio   {np.exp(np.median(lg)):6.2f}   "
          f"(1.00 = no bias)")
    print(f"  geometric mean           {np.exp(lg.mean()):6.2f}")
    print(f"  geometric sd             {np.exp(lg.std()):6.2f}x")
    for thr in (1.25, 1.5, 2.0, 3.0):
        n = int(((r <= thr) & (r >= 1 / thr)).sum())
        print(f"  within {thr:4.2f}x            {n:4d} of {len(r)}  "
              f"({100.0 * n / len(r):5.1f}%)")

    # Stratify by magnitude and by mixed-layer thickness. Both matter for how
    # the number can honestly be displayed: the low-PFT tail is where ratios go
    # wild, and thin mixed layers were the suspected cause of the scatter.
    print()
    for lo, hi, lbl in ((0, 50, "RAOB < 50 GW"), (50, 200, "RAOB 50-200 GW"),
                        (200, 1e9, "RAOB >= 200 GW")):
        sub = np.array([x["ratio"] for x in rows
                        if lo <= x["raob"] < hi and np.isfinite(x["ratio"])
                        and x["ratio"] > 0])
        if len(sub) >= 3:
            l = np.log(sub)
            n15 = int(((sub <= 1.5) & (sub >= 1 / 1.5)).sum())
            print(f"  {lbl:16s} n={len(sub):4d}  median {np.exp(np.median(l)):5.2f}  "
                  f"geo-sd {np.exp(l.std()):5.2f}x  within 1.5x {n15:3d} "
                  f"({100.0 * n15 / len(sub):4.1f}%)")
    for want_thin, lbl in ((True, "ML thin (<=2 lv)"), (False, "ML not thin")):
        sub = np.array([x["ratio"] for x in rows
                        if bool(x.get("thin")) is want_thin
                        and np.isfinite(x["ratio"]) and x["ratio"] > 0])
        if len(sub) >= 3:
            l = np.log(sub)
            n15 = int(((sub <= 1.5) & (sub >= 1 / 1.5)).sum())
            print(f"  {lbl:16s} n={len(sub):4d}  median {np.exp(np.median(l)):5.2f}  "
                  f"geo-sd {np.exp(l.std()):5.2f}x  within 1.5x {n15:3d} "
                  f"({100.0 * n15 / len(sub):4.1f}%)")
    print()

    for key, nm in (("zr", "z_fc"), ("dr", "dtheta_fc"), ("ur", "U_ML")):
        v = np.array([x[key] for x in rows], dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            print(f"  {nm:10s} median ratio  {np.exp(np.median(np.log(v))):6.2f}")

    orog_err = np.array([x["orog"] - x["elev"] for x in rows])
    print(f"\n  HRRR orography minus station elevation: "
          f"median {np.median(orog_err):+.0f} m, "
          f"max |err| {np.max(np.abs(orog_err)):.0f} m")
    print("  (z_fc is squared, so this is the terrain-representation error the")
    print("   production path carries at every fire, separate from the profile.)")

    worst = sorted(rows, key=lambda x: -abs(math.log(max(x["ratio"], 1e-9))))[:4]
    print("\n  Largest disagreements:")
    for w in worst:
        print(f"    {w['site']:22s} RAOB {w['raob']:8.1f}  HRRR {w['hrrr']:8.1f}  "
              f"ratio {w['ratio']:5.2f}   z_fc ratio {w['zr']:.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="YYYY-MM-DD (default: yesterday UTC)")
    ap.add_argument("--hour", type=int, choices=(0, 12),
                    help="synoptic hour; omit to run both")
    ap.add_argument("--days", type=int, default=1,
                    help="number of days back to cover (default 1)")
    ap.add_argument("--sites", default="",
                    help="comma-separated WMO ids; default all western sites")
    ap.add_argument("--workdir", default=".", help="scratch dir for GRIB subsets")
    args = ap.parse_args()

    sites = WESTERN_RAOB
    if args.sites:
        want = {int(s) for s in args.sites.split(",") if s.strip()}
        sites = {k: v for k, v in WESTERN_RAOB.items() if k in want}
        if not sites:
            ap.error("no known sites matched --sites")

    if args.date:
        base = dt.datetime.strptime(args.date, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc)
    else:
        base = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)

    hours = [args.hour] if args.hour is not None else [0, 12]
    all_rows = []
    for d in range(args.days):
        day = base - dt.timedelta(days=d)
        for h in hours:
            all_rows += run_cycle(day.replace(hour=h), sites, workdir=args.workdir)
    return summarise(all_rows)


if __name__ == "__main__":
    sys.exit(main())
