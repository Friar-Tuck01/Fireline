"""
Wind-slope alignment from HRRR's native 3 km wind field.

Replaces the Open-Meteo point-sampling version. Open-Meteo serves HRRR, but
only as point queries -- we sampled a 3 km field every 0.5 degrees and
interpolated between samples, which aliases badly in complex terrain (a
downslope windstorm confined to one canyon can fall between sample points).
NOAA's NOMADS GRIB filter hands over the actual field, server-side subset to
our box, free and without a key.

    python3 daily_wsa.py run
    python3 daily_wsa.py cycles     # what HRRR cycles are on the server

WHY u AND v RATHER THAN SPEED AND DIRECTION
    GRIB carries the wind as vector components, so

        W . grad(h) = u * dh/dx + v * dh/dy

    directly. No trigonometry, and no chance of the classic meteorological
    error of treating "direction the wind comes from" as "direction it is
    going". Those cancel out here because u and v already point the way the
    air is moving.

HOURLY, THEN THE EXTREME
    One HRRR cycle gives f00 onward in hourly steps. We evaluate alignment at
    every hour and keep the largest magnitude, sign preserved -- "the worst
    alignment today", which is the operational question. A daily-mean wind
    direction would average away a canyon that flips from lee to windward
    across the afternoon, which is exactly when it matters.

DEPENDENCIES
    Needs a GRIB2 reader. pygrib is tried first, cfgrib second; both are
    conda- and pip-installable. This is the heaviest dependency in the
    project and the reason the Open-Meteo path is kept as a fallback.
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.request

import numpy as np

import build_climatology as bc

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
OUT_DIR = os.path.join(bc.WORK, "daily")
GRIB_DIR = os.path.join(bc.WORK, "hrrr")

# HRRR runs hourly, but the 00/06/12/18Z cycles go out to f48 while the rest
# stop at f18. Using a synoptic cycle guarantees a full day of forecast hours
# from a single model run -- mixing cycles would splice together forecasts
# initialised at different times.
SYNOPTIC = (0, 6, 12, 18)
FORECAST_HOURS = 24
# NOMADS asks for a pause between scripted fetches. Honour it: this is a
# public service with no key and no quota, held up by people being decent.
FETCH_PAUSE = 10.0

# Thresholds in m/s of terrain-forced vertical velocity, anchored to the
# observed HRRR distribution rather than guessed:
#
#     |W.grad(h)| over land:  p50 0.19   p90 1.12   p99 2.85
#
# so 0.5 / 1.5 / 3.0 lands near the 82nd, 93rd and 99th percentiles -- a
# pyramid of roughly 18 / 6 / 1 percent of cells.
#
# These are 2x the values used for the gridMET version, and deliberately so.
# That layer used a daily-MEAN wind on a 4 km grid; this one takes the
# hourly MAXIMUM from a 3 km model, and both changes push the distribution
# up. Carrying the old thresholds over flagged 42% of the West with "strong"
# at 5.9%, which is not a warning, it is wallpaper.
WSA_BANDS = [
    ( 0.5, (250, 210,  70, 150), 'weak upslope'),
    ( 1.5, (240, 120,  30, 190), 'moderate upslope'),
    ( 3.0, (196,  16,  32, 220), 'strong upslope'),
    (-0.5, (150, 205, 240, 150), 'weak lee'),
    (-1.5, ( 70, 130, 220, 190), 'moderate lee'),
    (-3.0, (110,  50, 190, 220), 'strong lee'),
]


def _read_grib(path):
    """Return (u10, v10, lats, lons) from a GRIB2 file.

    Two readers because neither is reliably installable everywhere. pygrib is
    the simpler API; cfgrib ships with eccodes wheels that sometimes work
    where pygrib does not.
    """
    try:
        import pygrib
        with pygrib.open(path) as g:
            u = v = lats = lons = None
            for msg in g:
                sn = getattr(msg, "shortName", "")
                if sn in ("10u", "u10") or "U component" in str(msg):
                    u = msg.values.astype(np.float32)
                    lats, lons = msg.latlons()
                elif sn in ("10v", "v10") or "V component" in str(msg):
                    v = msg.values.astype(np.float32)
            if u is None or v is None:
                raise RuntimeError(f"no 10 m wind in {path}")
            return u, v, lats.astype(np.float32), lons.astype(np.float32)
    except ImportError:
        pass

    try:
        import cfgrib
    except ImportError:
        sys.exit("Needs a GRIB2 reader:\n"
                 "    conda install -c conda-forge pygrib\n"
                 "  or\n"
                 "    pip install cfgrib eccodes")
    ds = cfgrib.open_file(path)
    names = list(ds.variables)
    uk = next(k for k in names if k in ("u10", "10u"))
    vk = next(k for k in names if k in ("v10", "10v"))
    u = np.asarray(ds.variables[uk].data, dtype=np.float32)
    v = np.asarray(ds.variables[vk].data, dtype=np.float32)
    lats = np.asarray(ds.variables["latitude"].data, dtype=np.float32)
    lons = np.asarray(ds.variables["longitude"].data, dtype=np.float32)
    return u, v, lats, lons


def pick_cycle():
    """Most recent synoptic cycle old enough to have been published.

    HRRR files appear roughly 50-90 minutes after the cycle time, so anything
    newer than ~3 hours is a coin flip. Being conservative here costs a few
    hours of forecast lead and avoids a run that fails on missing files.
    """
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)
    for back in range(0, 30):
        t = now - dt.timedelta(hours=back)
        if t.hour in SYNOPTIC:
            return t.replace(minute=0, second=0, microsecond=0)
    raise RuntimeError("no synoptic cycle found")


def grib_url(cycle, fh, bounds):
    south, west, north, east = bounds
    return (f"{NOMADS}?dir=%2Fhrrr.{cycle:%Y%m%d}%2Fconus"
            f"&file=hrrr.t{cycle:%H}z.wrfsfcf{fh:02d}.grib2"
            f"&var_UGRD=on&var_VGRD=on&lev_10_m_above_ground=on"
            f"&subregion=&toplat={north}&leftlon={west}"
            f"&rightlon={east}&bottomlat={south}")


def fetch_hour(cycle, fh, bounds, tries=3):
    os.makedirs(GRIB_DIR, exist_ok=True)
    dest = os.path.join(GRIB_DIR, f"hrrr_{cycle:%Y%m%d%H}_f{fh:02d}.grib2")
    if os.path.exists(dest) and os.path.getsize(dest) > 5000:
        return dest
    url = grib_url(cycle, fh, bounds)
    tmp = dest + ".tmp"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "fireline-wsa/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            # The filter returns a short HTML error page rather than an HTTP
            # error when a file is missing, so check the GRIB magic number
            # instead of trusting the status code.
            if not body.startswith(b"GRIB"):
                raise RuntimeError(
                    f"not GRIB data (got {body[:80]!r}) — the cycle may not "
                    f"be published yet")
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == tries - 1:
                raise
            time.sleep(15)
    return None


def build_index(hlat, hlon, tlat, tlon, cache):
    """Nearest HRRR cell for every terrain cell.

    HRRR is on a Lambert Conformal grid, so there is no analytic row/column
    lookup -- the mapping has to be found by proximity. It depends only on
    two fixed grids, so it is computed once and cached; recomputing costs
    about a minute.
    """
    if os.path.exists(cache):
        return np.load(cache)
    from scipy.spatial import cKDTree
    print("   building HRRR -> terrain index (once; ~1 min)")
    tree = cKDTree(np.column_stack([hlat.ravel(), hlon.ravel()]))
    rows, cols = len(tlat), len(tlon)
    idx = np.zeros((rows, cols), dtype=np.int32)
    LO = np.repeat(tlon[None, :], 500, axis=0)
    for r0 in range(0, rows, 500):
        r1 = min(r0 + 500, rows)
        LA = np.repeat(tlat[r0:r1, None], cols, axis=1)
        q = np.column_stack([LA.ravel(), LO[:r1-r0].ravel()])
        _, ii = tree.query(q, workers=-1)
        idx[r0:r1] = ii.reshape(r1 - r0, cols)
        print(f"      rows {r0:>5}-{r1:>5}")
    np.save(cache, idx)
    return idx


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    t = _terrain()
    slope_deg = t["slope"].astype(np.float32)
    uphill = np.deg2rad(t["aspect"].astype(np.float32) * 360.0 / 255.0)
    valid = t["valid"]
    rows, cols = slope_deg.shape
    south, west, north, east = [float(x) for x in t["bounds"]]

    # Gradient components once. gx is east-positive, gy north-positive, so
    # they pair directly with GRIB's u and v.
    tan_s = np.tan(np.deg2rad(np.clip(slope_deg, 0, 60)))
    gx = (tan_s * np.sin(uphill)).astype(np.float32)
    gy = (tan_s * np.cos(uphill)).astype(np.float32)
    del tan_s, uphill

    cycle = pick_cycle()
    print(f"HRRR cycle {cycle:%Y-%m-%d %HZ}, f00-f{FORECAST_HOURS-1:02d}")

    tlat = np.linspace(north - 0.5/240, south + 0.5/240, rows)
    tlon = np.linspace(west + 0.5/240, east - 0.5/240, cols)
    idx = None
    best = np.zeros((rows, cols), dtype=np.float32)
    best_hour = np.zeros((rows, cols), dtype=np.int8)
    nh = 0

    for fh in range(FORECAST_HOURS):
        try:
            path = fetch_hour(cycle, fh, (south, west, north, east))
        except Exception as e:
            print(f"   f{fh:02d}: {e}")
            break
        u, v, hlat, hlon = _read_grib(path)
        if idx is None:
            print(f"   HRRR subset grid {u.shape} = {u.size:,} native cells "
                  f"at ~3 km")
            idx = build_index(hlat, hlon, tlat, tlon,
                              os.path.join(bc.WORK, "hrrr_index.npy"))
        uu = u.ravel()[idx]
        vv = v.ravel()[idx]
        w = uu * gx + vv * gy
        take = np.abs(w) > np.abs(best)
        best = np.where(take, w, best)
        best_hour = np.where(take, fh, best_hour).astype(np.int8)
        nh += 1
        if fh % 6 == 0:
            print(f"   f{fh:02d}  peak |W.grad(h)| so far "
                  f"{np.abs(best[valid]).max():.2f} m/s")
        os.remove(path)
        time.sleep(FETCH_PAUSE)

    if nh == 0:
        sys.exit("no HRRR hours retrieved")
    _render(best, best_hour, valid, rows, cols,
            (south, west, north, east), cycle, nh)


def _terrain():
    for c in (os.path.join(bc.WORK, "breakpoints", "terrain_450m.npz"),
              "terrain_450m.npz", os.path.join("data", "terrain_450m.npz")):
        if os.path.exists(c):
            return np.load(c)
    sys.exit("terrain_450m.npz not found — run build_terrain.py")


def _render(best, best_hour, valid, rows, cols, bounds, cycle, nh):
    south, west, north, east = bounds
    cls = np.zeros((rows, cols), dtype=np.int8)
    for i, (thr, _c, _n) in enumerate(WSA_BANDS, start=1):
        cls = np.where(best >= thr if thr > 0 else best <= thr, i, cls)

    from PIL import Image
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    for i, (_t, col, _n) in enumerate(WSA_BANDS, start=1):
        rgba[(cls == i) & valid] = col
    png = os.path.join(OUT_DIR, "wsa_latest.png")
    Image.fromarray(rgba, "RGBA").save(png, optimize=True)

    nland = int(valid.sum())
    counts = {n: int(((cls == i+1) & valid).sum())
              for i, (_t, _c, n) in enumerate(WSA_BANDS)}
    with open(os.path.join(OUT_DIR, "wsa_latest.json"), "w") as f:
        json.dump({
            "valid_date": cycle.date().isoformat(),
            "model_cycle_utc": cycle.isoformat().replace("+00:00", "Z"),
            "generated_utc": dt.datetime.now(dt.timezone.utc)
                               .isoformat().replace("+00:00", "Z"),
            "bounds": [[south, west], [north, east]],
            "bands": [n for _t, _c, n in WSA_BANDS],
            "colors": ["#%02X%02X%02X" % c[:3] for _t, c, _n in WSA_BANDS],
            "band_cell_counts": counts,
            "resolution_m": 450,
            "hours_evaluated": nh,
            "aggregation": "largest-magnitude forecast hour, sign preserved",
            "source": "W.grad(h); HRRR 3 km 10 m wind via NOAA NOMADS, "
                      "terrain from SRTM/3DEP",
        }, f, indent=2)

    print(f"\nWind-slope alignment, HRRR {cycle:%Y-%m-%d %HZ}, {nh} hours "
          f"({os.path.getsize(png)/1e6:.1f} MB)")
    for _t, _c, n in WSA_BANDS:
        print(f"   {n:>17}: {counts[n]:>9,} cells "
              f"{100*counts[n]/nland:>5.2f}%")
    ab = np.abs(best[valid])
    print(f"   |W.grad(h)| p50 {np.percentile(ab,50):.2f}  "
          f"p90 {np.percentile(ab,90):.2f}  p99 {np.percentile(ab,99):.2f}  "
          f"max {ab.max():.2f}")
    hrs, cnt = np.unique(best_hour[valid], return_counts=True)
    top = sorted(zip(cnt, hrs), reverse=True)[:5]
    print("   peak alignment forecast hour: "
          + ", ".join(f"f{int(h):02d} {100*c/nland:.0f}%" for c, h in top))


def cycles():
    """Probe which recent cycles are actually on the server."""
    t = _terrain()
    south, west, north, east = [float(x) for x in t["bounds"]]
    now = dt.datetime.now(dt.timezone.utc)
    print(f"now {now:%Y-%m-%d %H:%MZ}\n")
    for back in range(0, 30):
        c = (now - dt.timedelta(hours=back)).replace(
            minute=0, second=0, microsecond=0)
        if c.hour not in SYNOPTIC:
            continue
        url = grib_url(c, 0, (south, west, north, east))
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "fireline/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                print(f"  {c:%Y-%m-%d %HZ}  HTTP {r.status}")
        except Exception as e:
            print(f"  {c:%Y-%m-%d %HZ}  {e}")
        time.sleep(2)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "cycles": cycles}[cmd]()
