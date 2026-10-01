"""
hrrr_pft.py -- HRRR vertical profiles for PFT, by byte-range subset from AWS.

Downloads only the GRIB records it needs from the public HRRR archive on S3,
using the .idx inventory to work out byte offsets. A full wrfprs file is ~700 MB;
the subset this pulls (TMP/SPFH/HGT/UGRD/VGRD on 25 hPa isobaric levels plus a
few surface fields) is a small fraction of that.

Bucket:  noaa-hrrr-bdp-pds   (anonymous access, no AWS account needed)
Path:    hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfprsfFF.grib2
Index:   ...grib2.idx

----------------------------------------------------------------------------
Gotchas this file exists to handle
----------------------------------------------------------------------------
* .idx byte ranges. Each line gives a START offset only. A record's end is the
  NEXT line's offset minus one. The last record runs to EOF. Getting this wrong
  yields a truncated GRIB that pygrib may still open, with the final message
  silently missing.

* Isobaric levels BELOW GROUND. HRRR reports every isobaric level over the whole
  domain, including levels underneath terrain, where values are extrapolated
  fiction. At Mount Whitney the 1000, 975, 950... levels are all underground.
  Every level whose geopotential height is at or below the model orography must
  be dropped, or the mixed-layer average is computed over imaginary air.

* z_fc is measured above the FIRE GROUND and enters PFT squared. A 1 km
  elevation error moves PFT by 2-4x. This is the single largest avoidable error
  in the whole chain, which is why the elevation source is recorded per fire.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HRRR_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# Isobaric levels to pull, in hPa.
#
# Variable spacing, for two reasons.
#
# HEADROOM. The buoyancy test runs from the saturation point up to the -20 C
# level, so the profile must reach it. A first cut stopped at 400 hPa, which
# looked fine against a Sierra profile (-20 C near 510 hPa) but fails on the
# fires that matter most: a 40 C valley fire puts the -20 C level near 360 hPa
# and a hot desert fire near 330. Those would have come back "no solution"
# while every cool high-elevation fire worked -- a bias that would have looked
# like real signal. 300 hPa clears all of them.
#
# RESOLUTION WHERE IT COUNTS. 25 hPa spacing matters from the surface up
# through z_fc (typically 2-6 km, i.e. ~800-500 hPa), because that is where
# the mixed layer is diagnosed and where the saturation point lands. Above
# that only the environmental temperature is needed, for the buoyancy
# comparison, and 50 hPa resolves that fine.
#
# Net effect: 23 levels instead of 25, so a SMALLER download than the naive
# 25 hPa-throughout version, with 100 hPa more headroom.
LEVELS_HPA = (list(range(1000, 575, -25))    # 1000, 975, ... 600  (17 levels)
              + list(range(550, 275, -50)))  # 550, 500, ... 300   (6 levels)

# Per-level variables. SPFH is specific humidity, which is what the solver
# wants directly -- no RH-to-q conversion and no temperature dependence in the
# round trip.
LEVEL_VARS = ("TMP", "SPFH", "HGT", "UGRD", "VGRD")

# Surface fields. HGT:surface is the model orography, needed to reject
# below-ground levels.
SURFACE_RECORDS = (
    ("HGT", "surface"),
    ("PRES", "surface"),
)

# Near-surface records, added because the isobaric levels alone start too high.
#
# At 25 hPa spacing the first isobaric level above ground can sit 200+ m up, so
# the diagnosed mixed layer often contained only ONE OR TWO levels. Measured
# against radiosondes over 25 site-hours, those thin cases were both more
# biased and far noisier than the rest (median 0.75 and geo-sd 1.79x, against
# 0.87 and 1.30x when the layer held more than two levels).
#
# 2 m temperature and humidity with 10 m winds anchor the bottom of the layer.
# This ADDS INFORMATION rather than interpolating what is already there, which
# is the difference between a real fix and a cosmetic one.
NEAR_SURFACE_RECORDS = (
    ("TMP", "2 m above ground"),
    ("SPFH", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
)

USER_AGENT = "fireline-pft/1.0 (+https://github.com/Friar-Tuck01/Fireline)"


# ---------------------------------------------------------------------------
# Cycle selection
# ---------------------------------------------------------------------------
def cycle_candidates(target_valid_hour_utc: int = 21,
                     now: dt.datetime | None = None,
                     max_fxx: int = 18,
                     n: int = 6):
    """
    Candidate (cycle_datetime, forecast_hour) pairs valid at a target hour.

    Default target is 21Z -- roughly 2 pm PDT / 3 pm MDT, the peak fire-weather
    hour. The paper's own worked examples and the Utah PFT product both focus on
    the afternoon, and Fireline's validated cases were afternoon soundings; the
    one case where the solver ran high was a 0600 nocturnal-inversion profile.

    Candidates are returned newest first. HRRR cycles post roughly 45-90 minutes
    after the hour, so cycles younger than ~2 h are skipped rather than 404ing.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    # Walk back from the most recent plausibly-complete cycle.
    newest = (now - dt.timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    for back in range(0, 48):
        cyc = newest - dt.timedelta(hours=back)
        # Forecast hour that lands on the target valid hour.
        fxx = (target_valid_hour_utc - cyc.hour) % 24
        if fxx > max_fxx:
            continue
        out.append((cyc, fxx))
        if len(out) >= n:
            break
    return out


def hrrr_urls(cycle: dt.datetime, fxx: int):
    stem = (f"{HRRR_BASE}/hrrr.{cycle:%Y%m%d}/conus/"
            f"hrrr.t{cycle:%H}z.wrfprsf{fxx:02d}.grib2")
    return stem, stem + ".idx"


# ---------------------------------------------------------------------------
# .idx handling
# ---------------------------------------------------------------------------
def _get(url, timeout=60, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_idx(text: str):
    """
    Parse a wgrib-style .idx inventory.

    Lines look like:
        12:5551234:d=2026092712:TMP:500 mb:9 hour fcst:

    Returns a list of dicts with num, offset, var, level, and (filled in
    afterwards) end.
    """
    recs = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        try:
            num, offset = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        recs.append({"num": num, "offset": offset, "date": parts[2],
                     "var": parts[3], "level": parts[4], "fcst": parts[5]})
    recs.sort(key=lambda r: r["offset"])
    # A record ends one byte before the next one starts. The last is open-ended.
    for i, r in enumerate(recs):
        r["end"] = recs[i + 1]["offset"] - 1 if i + 1 < len(recs) else None
    return recs


def select_records(recs, levels_hpa=LEVELS_HPA, level_vars=LEVEL_VARS,
                   surface=SURFACE_RECORDS, near_surface=NEAR_SURFACE_RECORDS):
    """Pick the records we need, preserving file order."""
    want_levels = {f"{p} mb" for p in levels_hpa}
    want_sfc = set(surface) | set(near_surface)
    keep = []
    for r in recs:
        if r["var"] in level_vars and r["level"] in want_levels:
            keep.append(r)
        elif (r["var"], r["level"]) in want_sfc:
            keep.append(r)
    return keep


def merge_ranges(recs, max_gap=2_000_000):
    """
    Merge nearby byte ranges into runs.

    One HTTP request per GRIB record would mean ~130 requests. The records we
    want are largely contiguous in the file, so merging anything separated by
    less than max_gap bytes collapses that to a handful of requests at the cost
    of a little wasted download.
    """
    if not recs:
        return []
    runs = []
    start = recs[0]["offset"]
    end = recs[0]["end"]
    for r in recs[1:]:
        if end is None:
            break
        if r["offset"] - end <= max_gap:
            end = r["end"] if r["end"] is not None else None
            if end is None:
                break
        else:
            runs.append((start, end))
            start, end = r["offset"], r["end"]
    runs.append((start, end))
    return runs


def download_subset(grib_url, runs, out_path, timeout=180, attempts=3):
    """
    Fetch each byte range and concatenate into a local GRIB2 file.

    Retries each range. Large transfers from S3 occasionally cut out mid-stream
    (IncompleteRead), and without a retry one truncated range loses the whole
    cycle -- which cost a validation cycle once. A partial range is discarded
    and re-requested rather than appended, since a short read would corrupt
    the GRIB with no error at write time.
    """
    total = 0
    with open(out_path, "wb") as f:
        for (start, end) in runs:
            rng = f"bytes={start}-" + ("" if end is None else str(end))
            expect = None if end is None else (end - start + 1)
            data = None
            last = None
            for attempt in range(attempts):
                try:
                    got = _get(grib_url, timeout=timeout, headers={"Range": rng})
                    if expect is not None and len(got) != expect:
                        raise OSError(f"short read: {len(got)} of {expect} bytes")
                    data = got
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(1.5 * (attempt + 1))
            if data is None:
                raise RuntimeError(
                    f"byte range {rng} failed after {attempts} attempts: {last}")
            f.write(data)
            total += len(data)
    return total


# ---------------------------------------------------------------------------
# GRIB -> arrays
# ---------------------------------------------------------------------------
# ecCodes short name -> the NCEP name used in the .idx and everywhere below.
#
# THIS MAPPING IS THE WHOLE POINT. The .idx inventory uses NCEP abbreviations
# (HGT, TMP, SPFH, UGRD, VGRD), so those are what select_records matches on.
# But pygrib exposes msg.shortName from ecCodes, which uses entirely different
# names (gh, t, q, u, v), and surface geopotential height comes back as 'orog'
# rather than any spelling of HGT. Selecting the right records and then filing
# them under names nothing looks up produces a bare KeyError a long way from
# the cause.
ECCODES_TO_NCEP = {
    "gh": "HGT", "orog": "HGT", "h": "HGT", "hgt": "HGT",
    "t": "TMP", "tmp": "TMP",
    "q": "SPFH", "spfh": "SPFH",
    "u": "UGRD", "ugrd": "UGRD", "10u": "UGRD",
    "v": "VGRD", "vgrd": "VGRD", "10v": "VGRD",
    "sp": "PRES", "pres": "PRES",
}

# Fallback on the long name, for builds whose shortName is 'unknown'.
LONGNAME_TO_NCEP = {
    "geopotential height": "HGT",
    "orography": "HGT",
    "temperature": "TMP",
    "specific humidity": "SPFH",
    "u component of wind": "UGRD",
    "v component of wind": "VGRD",
    "surface pressure": "PRES",
    "pressure": "PRES",
}


def canonical_var(msg):
    """
    NCEP-style variable name for a pygrib message, or None if unrecognised.

    Tries ecCodes shortName first, then the long name. Returning None rather
    than guessing keeps an unexpected record out of the profile instead of
    letting it masquerade as one we wanted.
    """
    sn = str(getattr(msg, "shortName", "") or "").strip().lower()
    if sn in ECCODES_TO_NCEP:
        return ECCODES_TO_NCEP[sn]
    ln = str(getattr(msg, "name", "") or "").strip().lower()
    if ln in LONGNAME_TO_NCEP:
        return LONGNAME_TO_NCEP[ln]
    # Some builds spell it straight through.
    up = sn.upper()
    if up in ("HGT", "TMP", "SPFH", "UGRD", "VGRD", "PRES"):
        return up
    return None


def inspect_grib(grib_path, limit=40):
    """Print what a GRIB file actually contains. For diagnosing name mismatches."""
    import pygrib
    print(f"{'shortName':<12}{'typeOfLevel':<18}{'level':>8}  {'->':<6}{'name'}")
    print("-" * 88)
    with pygrib.open(grib_path) as gr:
        for i, msg in enumerate(gr):
            if i >= limit:
                print(f"... ({i}+ messages)")
                break
            print(f"{str(msg.shortName):<12}{str(msg.typeOfLevel):<18}"
                  f"{str(msg.level):>8}  {str(canonical_var(msg)):<6}{msg.name}")


def read_fields(grib_path):
    """
    Read the subset into arrays.

    Returns (fields, lats, lons) where fields maps ("TMP", 500) -> 2D array and
    ("HGT", "surface") -> 2D array.
    """
    try:
        import pygrib
    except ImportError:
        raise SystemExit(
            "pygrib is required to read HRRR GRIB2. In the fireline env:\n"
            "    conda install -c conda-forge pygrib")

    fields, lats, lons = {}, None, None
    seen = []
    with pygrib.open(grib_path) as gr:
        for msg in gr:
            if lats is None:
                lats, lons = msg.latlons()
            try:
                lvl_type = msg.typeOfLevel
            except Exception:  # noqa: BLE001
                lvl_type = ""
            name = canonical_var(msg)
            seen.append((getattr(msg, "shortName", "?"), getattr(msg, "name", "?"),
                         lvl_type, getattr(msg, "level", "?"), name))
            if name is None:
                continue
            vals = np.asarray(msg.values, dtype=float)
            if lvl_type == "isobaricInhPa":
                fields[(name, int(msg.level))] = vals
            elif lvl_type == "surface":
                fields[(name, "surface")] = vals
            elif lvl_type == "heightAboveGround" and int(msg.level) in (2, 10):
                fields[(name, f"{int(msg.level)}m")] = vals

    if lats is None:
        raise ValueError(f"No GRIB messages found in {grib_path}")

    # Fail loudly, and say what WAS in the file. Without this the failure is a
    # bare KeyError that gives no hint the names simply differ.
    if ("HGT", "surface") not in fields:
        inv = "\n".join(f"    shortName={s!r} name={n!r} level={lt}/{lv} -> {c!r}"
                        for s, n, lt, lv, c in seen[:25])
        raise ValueError(
            "Surface orography ('HGT','surface') missing after decoding. pygrib "
            "reports ecCodes short names, which differ from the NCEP names in "
            "the .idx. First messages found:\n" + inv)
    # HRRR longitudes come back 0-360; Fireline works in -180..180.
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    return fields, lats, lons


def build_index(lats, lons):
    """
    KD-tree over the HRRR grid for nearest-neighbour lookup.

    Built in 3D cartesian on the unit sphere rather than in (lat, lon), so it
    is correct without worrying about convergence of meridians or the Lambert
    projection.
    """
    from scipy.spatial import cKDTree
    la, lo = np.radians(lats.ravel()), np.radians(lons.ravel())
    xyz = np.column_stack([np.cos(la) * np.cos(lo),
                           np.cos(la) * np.sin(lo),
                           np.sin(la)])
    return cKDTree(xyz), lats.shape


def nearest_indices(tree, shape, fire_lats, fire_lons):
    la, lo = np.radians(np.asarray(fire_lats)), np.radians(np.asarray(fire_lons))
    xyz = np.column_stack([np.cos(la) * np.cos(lo),
                           np.cos(la) * np.sin(lo),
                           np.sin(la)])
    _, flat = tree.query(xyz, k=1)
    return np.unravel_index(flat, shape)


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------
def profile_at(fields, iy, ix, levels_hpa=LEVELS_HPA, min_levels=8):
    """
    Build a surface-first profile at one grid point.

    Levels at or below the model orography are DROPPED. HRRR publishes isobaric
    values underneath terrain -- they are extrapolated, not observed, and on a
    Sierra or Rockies fire they can be several hundred hPa of pure fiction. A
    mixed layer averaged over them is meaningless.

    Returns (dict of arrays, orography_m) or (None, orography_m) if too few
    levels survive.
    """
    orog = float(fields[("HGT", "surface")][iy, ix])

    p, z, T, q, u, v = [], [], [], [], [], []

    # Near-surface level first, when available. Without it the profile starts
    # at the lowest isobaric level above ground -- up to 250 m up -- and the
    # mixed-layer average has almost nothing to work with.
    try:
        p_sfc_pa = float(fields[("PRES", "surface")][iy, ix])
        t2 = float(fields[("TMP", "2m")][iy, ix])
        q2 = float(fields[("SPFH", "2m")][iy, ix])
        u10 = float(fields[("UGRD", "10m")][iy, ix])
        v10 = float(fields[("VGRD", "10m")][iy, ix])
        if all(np.isfinite(x) for x in (p_sfc_pa, t2, q2, u10, v10)) and p_sfc_pa > 1e4:
            p.append(p_sfc_pa / 100.0)   # GRIB2 pressure is Pa
            z.append(orog + 2.0)
            T.append(t2)
            q.append(max(q2, 1e-8) * 1000.0)
            # 10 m winds paired with 2 m thermodynamics: standard practice, and
            # the 8 m offset is far below the scales that matter here.
            u.append(u10)
            v.append(v10)
    except KeyError:
        pass  # older subset without the near-surface records
    for lev in levels_hpa:
        try:
            hgt = float(fields[("HGT", lev)][iy, ix])
        except KeyError:
            continue
        if not np.isfinite(hgt) or hgt <= orog + 1.0:
            continue  # underground, or sitting exactly on the surface
        if z and hgt <= z[-1] + 1.0:
            continue  # at or below the near-surface level already added
        try:
            t = float(fields[("TMP", lev)][iy, ix])
            sh = float(fields[("SPFH", lev)][iy, ix])
            uu = float(fields[("UGRD", lev)][iy, ix])
            vv = float(fields[("VGRD", lev)][iy, ix])
        except KeyError:
            continue
        if not all(np.isfinite(x) for x in (t, sh, uu, vv)):
            continue
        p.append(float(lev))
        z.append(hgt)
        T.append(t)
        q.append(max(sh, 1e-8) * 1000.0)  # kg/kg -> g/kg
        u.append(uu)
        v.append(vv)

    if len(p) < min_levels:
        return None, orog
    return {"p_hpa": np.array(p), "z_m": np.array(z), "T_k": np.array(T),
            "q_gkg": np.array(q), "u_ms": np.array(u), "v_ms": np.array(v)}, orog


def fetch_hrrr_subset(workdir=".", target_valid_hour_utc=21, verbose=True,
                      keep=False):
    """
    Find an available HRRR cycle, download the subset, and read it.

    Returns (fields, lats, lons, meta). Tries successively older cycles if the
    newest is not posted yet.
    """
    last_err = None
    for cycle, fxx in cycle_candidates(target_valid_hour_utc):
        grib_url, idx_url = hrrr_urls(cycle, fxx)
        try:
            if verbose:
                print(f"  trying {cycle:%Y-%m-%d %H}Z f{fxx:02d} ...", flush=True)
            idx_text = _get(idx_url, timeout=45).decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_err = f"{cycle:%Y%m%d %H}Z f{fxx:02d}: {e}"
            continue

        recs = parse_idx(idx_text)
        want = select_records(recs)
        if len(want) < 20:
            last_err = (f"{cycle:%Y%m%d %H}Z f{fxx:02d}: inventory had only "
                        f"{len(want)} matching records")
            continue

        runs = merge_ranges(want)
        out_path = os.path.join(workdir, f"hrrr_pft_{cycle:%Y%m%d%H}_f{fxx:02d}.grib2")

        # Reuse an existing subset for this exact cycle and forecast hour.
        # The download is ~170 MB; re-fetching it to debug something downstream
        # is pure waste. Delete the file, or pass --no-cache, to force a refetch.
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
            nbytes = os.path.getsize(out_path)
            if verbose:
                print(f"    reusing cached subset {out_path} "
                      f"({nbytes / 1e6:.1f} MB)", flush=True)
        else:
            if verbose:
                print(f"    {len(want)} records in {len(runs)} byte range(s)",
                      flush=True)
            nbytes = download_subset(grib_url, runs, out_path)
            if verbose:
                print(f"    downloaded {nbytes / 1e6:.1f} MB -> {out_path}",
                      flush=True)

        try:
            fields, lats, lons = read_fields(out_path)
        except Exception:
            # Keep the file when decoding fails -- otherwise the next attempt
            # re-downloads 170 MB just to hit the same error. Inspect it with:
            #     python hrrr_pft.py --inspect <file>
            print(f"    decode failed; subset kept at {out_path}\n"
                  f"    inspect it with:  python hrrr_pft.py --inspect {out_path}",
                  file=sys.stderr)
            raise
        if not keep:
            try:
                os.remove(out_path)
            except OSError:
                pass
        meta = {
            "model": "HRRR",
            "cycle_utc": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
            "forecast_hour": fxx,
            "valid_utc": (cycle + dt.timedelta(hours=fxx)).strftime("%Y-%m-%dT%H:00:00Z"),
            "grib_url": grib_url,
            "records": len(want),
            "bytes": nbytes,
        }
        return fields, lats, lons, meta

    raise RuntimeError(f"No HRRR cycle available. Last error: {last_err}")


if __name__ == "__main__":
    # --inspect <file> dumps what a downloaded subset actually contains. Use it
    # whenever a variable comes back missing: the names pygrib reports are not
    # the names in the .idx.
    if len(sys.argv) > 2 and sys.argv[1] == "--inspect":
        inspect_grib(sys.argv[2])
        sys.exit(0)

    # Smoke test: fetch a subset and print one profile at a fixed point.
    # keep=True so a decode problem can be inspected without re-downloading.
    fields, lats, lons, meta = fetch_hrrr_subset(keep=True)
    print(json.dumps(meta, indent=2))
    tree, shape = build_index(lats, lons)
    iy, ix = nearest_indices(tree, shape, [39.5], [-120.5])
    prof, orog = profile_at(fields, int(iy[0]), int(ix[0]))
    print(f"orography {orog:.0f} m; "
          f"{0 if prof is None else len(prof['p_hpa'])} usable levels")
    if prof is not None:
        for i in range(min(6, len(prof["p_hpa"]))):
            print(f"  {prof['p_hpa'][i]:6.0f} hPa  {prof['z_m'][i]:7.0f} m  "
                  f"{prof['T_k'][i] - 273.15:6.1f} C  {prof['q_gkg'][i]:5.2f} g/kg")
    sys.exit(0)
