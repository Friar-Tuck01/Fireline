"""
Daily SFDI product: step the state forward and publish today's map.

Reads the persisted NFDRS state, pulls however many days of gridMET have
appeared since the last run, walks the engine forward, converts ERC and BI to
percentiles through the climatology lookup tables, and writes a PNG overlay
plus a small JSON manifest.

    python3 daily_sfdi.py run          # fetch, advance, classify, render
    python3 daily_sfdi.py status

WHY THREDDS AND NOT THE ANNUAL FILES. The climatology build pulled 350 MB
full-CONUS annual files because it needed every day of every year. A daily
job needs a handful of days over the western US, so it asks THREDDS for
exactly that -- a few MB instead of ~2.5 GB. Same data, three orders of
magnitude less of it.

WHY THE STATE MATTERS. ERC is a build-up index. If the state is lost, today's
value cannot be recomputed from today's weather; it would take months of
spin-up to become trustworthy again. The state file is the one artifact here
that is genuinely irreplaceable on a short timescale, and whatever runs this
job has to persist it between runs.

GRIDMET LAGS. Observations are typically several days behind real time. This
job publishes the most recent day gridMET actually has, and the manifest
records that date explicitly -- the map must never imply it is showing today
when it is showing last Thursday.
"""

import datetime as dt
import json
import os
import pickle
import sys
import time
import urllib.request

import numpy as np

import build_climatology as bc

THREDDS = "http://thredds.northwestknowledge.net:8080/thredds"
OUT_DIR = os.path.join(bc.WORK, "daily")
MAX_CATCHUP_DAYS = 30    # refuse to silently paper over a long outage
# Raised temporarily to 400 for the one-time jump from the climatology's end
# (2025-12-31) to the present. Back at 30 now: in steady state gridMET lags
# 2-4 days, so being a month behind means the job has been failing quietly
# and should say so rather than grinding through a season's backlog.

# Class colors, RGBA. Alpha rises with severity so Low barely tints the
# basemap and Severe is unmissable -- on a fire map the top classes are what
# people are looking for, and a uniformly opaque overlay buries the terrain.
CLASS_COLORS = [
    (0x3A, 0x7D, 0x5F, 60),    # Low
    (0xD9, 0xC8, 0x4A, 110),   # Moderate
    (0xE0, 0x8A, 0x2E, 150),   # High
    (0xC1, 0x34, 0x2A, 190),   # Very High
    (0x8B, 0x2F, 0xB0, 225),   # Severe
]
CLASS_NAMES = ["Low", "Moderate", "High", "Very High", "Severe"]


def _lookup():
    for cand in (os.path.join(bc.WORK, "breakpoints", "sfdi_lookup.npz"),
                 "sfdi_lookup.npz", os.path.join("data", "sfdi_lookup.npz")):
        if os.path.exists(cand):
            return np.load(cand)
    sys.exit("sfdi_lookup.npz not found -- run build_breakpoints.py")


# The climatology and catch-up scripts track progress by YEAR ("next year to
# process"), which is the right granularity when you process whole years. A
# daily job advances a few days at a time, so it needs a DATE. Rather than
# overload the year field with a half-meaning -- the kind of ambiguity that
# eventually puts the state a day off with nothing to reveal it -- the daily
# job keeps its own marker and treats the year field as read-only.
def _last_date_path():
    return os.path.join(bc.META, "last_date.json")


def _last_processed():
    p = _last_date_path()
    if os.path.exists(p):
        with open(p) as f:
            return dt.date.fromisoformat(json.load(f)["last_date"])
    (_, next_year), _ = _state()
    return dt.date(next_year - 1, 12, 31)


def _set_last_processed(d):
    with open(_last_date_path(), "w") as f:
        json.dump({"last_date": d.isoformat()}, f)


def _state():
    sp = os.path.join(bc.META, "state.pkl")
    if not os.path.exists(sp):
        sys.exit("no state.pkl -- run build_climatology then catchup")
    with open(sp, "rb") as f:
        return pickle.load(f), sp


def fetch_days(var, start, end, tries=5):
    """Pull one variable over the western box for a date range."""
    fname = f"agg_met_{var}_1979_CurrentYear_CONUS.nc"
    url = (f"{THREDDS}/ncss/{fname}?var={bc.VARNAME[var]}"
           f"&north={bc.NORTH}&south={bc.SOUTH}"
           f"&west={bc.WEST}&east={bc.EAST}"
           f"&time_start={start}T00:00:00Z&time_end={end}T00:00:00Z"
           f"&accept=netcdf")
    dest = os.path.join(bc.RAW, f"daily_{var}.nc")
    tmp = dest + ".tmp"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "fireline-sfdi/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(tmp, "wb") as f:
                f.write(r.read())
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            wait = min(30, 2 ** attempt)
            print(f"   retry {attempt+1}/{tries} {var}: {e}; {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {var} {start}..{end}")


def _align_to_grid(arr, ds_lats, ds_lons, our_lats, our_lons):
    """Cut a THREDDS response down to exactly the climatology's grid.

    NCSS treats the bounding box as INCLUSIVE and returns any cell that
    touches it, so the same north/south/east/west values that produced a
    442-row subset from the annual files come back as 443 rows here. That is
    a one-row difference and it is fatal: the state arrays are indexed by
    flat cell position, so an extra row shifts every cell after the first
    one and would silently attribute Oregon's weather to California.

    Rather than trim by assumption, match on the coordinates themselves.
    gridMET is a fixed grid, so every one of our cells has an exact
    counterpart in whatever the server sends.
    """
    ilat = np.abs(ds_lats[None, :] - our_lats[:, None]).argmin(axis=1)
    ilon = np.abs(ds_lons[None, :] - our_lons[:, None]).argmin(axis=1)
    dlat = np.abs(ds_lats[ilat] - our_lats).max()
    dlon = np.abs(ds_lons[ilon] - our_lons).max()
    if dlat > 1e-3 or dlon > 1e-3:
        raise ValueError(
            f"returned grid does not line up with the climatology grid "
            f"(worst offset {dlat:.5f} lat, {dlon:.5f} lon). Refusing to "
            f"guess -- an unaligned grid silently mismatches cells.")
    return arr[:, ilat, :][:, :, ilon]


def run():
    import netCDF4

    (st, next_year), sp = _state()
    last_done = _last_processed()
    start = last_done + dt.timedelta(days=1)
    today = dt.date.today()
    # gridMET publishes with a lag; ask a bit past what we expect to exist and
    # take whatever comes back.
    end = today
    ndays_wanted = (end - start).days + 1
    if ndays_wanted <= 0:
        print("state is already current")
        return
    if ndays_wanted > MAX_CATCHUP_DAYS:
        sys.exit(
            f"{ndays_wanted} days behind ({start}..{end}). That is more than "
            f"MAX_CATCHUP_DAYS={MAX_CATCHUP_DAYS}. Fetching that much through "
            f"THREDDS is slow and something has probably gone wrong -- use "
            f"catchup.py for long gaps.")

    print(f"advancing {start} -> {end} (up to {ndays_wanted} days)")
    our_lats = np.load(os.path.join(bc.META, "lat.npy"))[:, 0]
    our_lons = np.load(os.path.join(bc.META, "lons.npy"))
    data, dates = {}, None
    for v in bc.MET_VARS:
        p = fetch_days(v, start.isoformat(), end.isoformat())
        with netCDF4.Dataset(p) as ds:
            vname = bc._data_var(ds, p)
            arr = ds.variables[vname][:]
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            arr = np.asarray(arr, dtype=np.float32)
            lo, hi = bc.PHYS[v]
            fin = np.isfinite(arr)
            nbad = int((fin & ((arr < lo) | (arr > hi))).sum())
            if fin.any() and nbad / max(int(fin.sum()), 1) > bc.BAD_FRACTION_LIMIT:
                sys.exit(f"{v}: {nbad:,} values outside [{lo},{hi}] -- "
                         f"THREDDS may not be applying scale_factor. Stop.")
            arr = np.clip(arr, lo, hi)
            arr = _align_to_grid(arr, np.asarray(ds.variables["lat"][:]),
                                 np.asarray(ds.variables["lon"][:]),
                                 our_lats, our_lons)
            data[v] = arr
            if dates is None:
                tv = ds.variables["day"]
                base = dt.date(1900, 1, 1)
                dates = [base + dt.timedelta(days=int(d)) for d in tv[:]]
        os.remove(p)
        print(f"   {v}: {data[v].shape}  (aligned to climatology grid)")

    print(f"gridMET has through {dates[-1]}")

    lat2d = np.load(os.path.join(bc.META, "lat.npy"))
    lat_flat = lat2d.ravel().astype(np.float64)
    shape2d = lat2d.shape

    erc_last = bi_last = None
    for i, d in enumerate(dates):
        jday = d.timetuple().tm_yday
        srad = np.nan_to_num(data["srad"][i].ravel(), nan=200.0)
        sow = bc.n.state_of_weather_from_srad(
            srad, bc._clear_sky(lat_flat, jday))
        out = bc.n.step(st, dict(
            tmax_f=_kf(data["tmmx"][i]), tmin_f=_kf(data["tmmn"][i]),
            rhmax=np.nan_to_num(data["rmax"][i].ravel(), nan=50.0),
            rhmin=np.nan_to_num(data["rmin"][i].ravel(), nan=25.0),
            tobs_f=_kf(data["tmmx"][i]),
            rhobs=np.nan_to_num(data["rmin"][i].ravel(), nan=25.0),
            sow=sow,
            ppt_in=np.nan_to_num(data["pr"][i].ravel(), nan=0.0) / 25.4,
            ws_mph=np.nan_to_num(data["vs"][i].ravel(), nan=4.0)
            * 0.914 * 2.23694,
            lat=lat_flat, jday=jday))
        erc_last, bi_last = out["erc"], out["bi"]
        print(f"   stepped {d}")

    # Persist state and the exact date it now reflects. Written together and
    # state first: if the process dies between them the next run re-processes
    # a day, which is harmless, rather than skipping one, which would leave a
    # permanent gap in a build-up index.
    with open(sp, "wb") as f:
        pickle.dump((st, dates[-1].year + 1), f)
    _set_last_processed(dates[-1])

    sfdi = classify(erc_last, bi_last)
    render(sfdi, shape2d, dates[-1])


def classify(erc, bi):
    """ERC and BI -> percentiles -> product -> class 0..4.

    p  = ERC' x BI'      (Jolly et al. 2019)
    p' = percentile of p, compared against the stored class thresholds.
    """
    lut = _lookup()
    erc_lut, bi_lut, p_thresh = lut["erc_lut"], lut["bi_lut"], lut["p_thresh"]
    n = erc_lut.shape[0]
    cells = np.arange(n)
    ep = erc_lut[cells, np.clip(erc, 0, erc_lut.shape[1] - 1).astype(int)]
    bp = bi_lut[cells, np.clip(bi, 0, bi_lut.shape[1] - 1).astype(int)]
    p = ep.astype(np.float32) * bp.astype(np.float32)
    # digitize against this cell's own thresholds -- the whole point of a
    # per-cell climatology is that 90 in the Great Basin is not 90 in Montana.
    cls = np.zeros(n, dtype=np.int8)
    for k in range(4):
        cls += (p > p_thresh[:, k]).astype(np.int8)
    return cls


def render(cls, shape2d, date):
    try:
        from PIL import Image
    except ImportError:
        sys.exit("needs Pillow:  conda install -y pillow")

    os.makedirs(OUT_DIR, exist_ok=True)
    cc = np.load(os.path.join(bc.META, "climate_class.npy"))
    land = np.isfinite(cc).ravel()

    rgba = np.zeros((cls.size, 4), dtype=np.uint8)
    for k, col in enumerate(CLASS_COLORS):
        rgba[(cls == k) & land] = col
    img = Image.fromarray(rgba.reshape(shape2d + (4,)), "RGBA")
    png = os.path.join(OUT_DIR, "sfdi_latest.png")
    img.save(png, optimize=True)

    lats = np.load(os.path.join(bc.META, "lat.npy"))
    lons = np.load(os.path.join(bc.META, "lons.npy"))
    counts = {CLASS_NAMES[k]: int(((cls == k) & land).sum()) for k in range(5)}
    nland = int(land.sum())
    manifest = {
        "valid_date": date.isoformat(),
        "generated_utc": dt.datetime.now(dt.timezone.utc)
                         .isoformat().replace("+00:00", "Z"),
        # Leaflet imageOverlay wants [[south, west], [north, east]].
        "bounds": [[float(lats.min()), float(lons.min())],
                   [float(lats.max()), float(lons.max())]],
        "shape": list(shape2d),
        "classes": CLASS_NAMES,
        "colors": ["#%02X%02X%02X" % c[:3] for c in CLASS_COLORS],
        "class_cell_counts": counts,
        "class_fractions": {k: round(v / nland, 4) for k, v in counts.items()},
        "source": "gridMET meteorology; NFDRS 1978 Fuel Model G; "
                  "SFDI after Jolly et al. 2019",
        "climatology": "1979-2017",
    }
    with open(os.path.join(OUT_DIR, "sfdi_latest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nvalid {date}   {png}  ({os.path.getsize(png)/1000:.0f} KB)")
    for k in range(5):
        frac = counts[CLASS_NAMES[k]] / nland
        print(f"   {CLASS_NAMES[k]:<10} {counts[CLASS_NAMES[k]]:>8,} cells "
              f"{100*frac:>5.1f}%")
    print("\nClimatological expectation is 60/20/10/7/3. A single day should")
    print("NOT match that -- a quiet winter day is nearly all Low, and a bad")
    print("August day can be a third Very High or worse. That spread is the")
    print("signal; if every day comes out near 60/20/10/7/3 the percentile")
    print("lookup is being applied wrong.")


def _kf(arr):
    return np.nan_to_num(arr.ravel(), nan=288.0) * 9 / 5 - 459.67


def status():
    (st, next_year), _ = _state()
    print(f"state reflects weather through {_last_processed()}")
    behind = (dt.date.today() - _last_processed()).days
    print(f"  {behind} days behind today "
          f"({'normal -- gridMET lags' if behind <= 8 else 'stale'})")
    p = os.path.join(OUT_DIR, "sfdi_latest.json")
    if os.path.exists(p):
        with open(p) as f:
            m = json.load(f)
        print(f"last published: {m['valid_date']} "
              f"(generated {m['generated_utc']})")
    else:
        print("nothing published yet")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "status": status}[cmd]()
