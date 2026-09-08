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


def fetch_year_slice(var, start, end):
    """Fallback: pull the bulk annual file and slice the days we need.

    THREDDS runs on port 8080 and is the single point of failure for this
    whole job -- a run died with "Connection refused" on all five retries
    while the previous day's run had succeeded against the same URL. The
    bulk annual files at northwestknowledge.net are plain HTTPS on 443 and
    have 39 years of proven reliability behind them (the entire climatology
    came through that path).

    Much heavier -- ~350 MB per variable versus a few MB for a bbox query --
    so this is strictly a fallback, never the first choice.
    """
    import netCDF4
    s_d = dt.date.fromisoformat(start)
    e_d = dt.date.fromisoformat(end)
    frames, days = [], []
    for year in range(s_d.year, e_d.year + 1):
        path = bc.download(var, year)
        with netCDF4.Dataset(path) as ds:
            vname = bc._data_var(ds, path)
            ilat, ilon, _, _ = bc._subset_index(ds)
            base = dt.date(1900, 1, 1)
            tv = np.asarray(ds.variables["day"][:])
            dates = [base + dt.timedelta(days=int(t)) for t in tv]
            keep = [i for i, d in enumerate(dates) if s_d <= d <= e_d]
            if not keep:
                continue
            arr = bc._read(ds, vname, slice(keep[0], keep[-1] + 1), ilat, ilon)
            frames.append(arr)
            days.extend(dates[keep[0]:keep[-1] + 1])
        os.remove(path)
    if not frames:
        raise RuntimeError(f"no {var} data for {start}..{end}")
    return np.concatenate(frames, axis=0), days


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
    use_fallback = False
    for v in bc.MET_VARS:
        if use_fallback:
            arr, dd = fetch_year_slice(v, start.isoformat(), end.isoformat())
            data[v] = np.clip(arr, *bc.PHYS[v])
            if dates is None:
                dates = dd
            print(f"   {v}: {data[v].shape}  (bulk annual file)")
            continue
        try:
            p = fetch_days(v, start.isoformat(), end.isoformat())
        except Exception as e:
            # Switch every remaining variable to the fallback too. Mixing
            # sources within one run would risk pairing days fetched two
            # different ways, and the grids must line up exactly.
            print(f"   THREDDS unavailable for {v} ({e}); "
                  f"falling back to bulk annual files")
            use_fallback = True
            arr, dd = fetch_year_slice(v, start.isoformat(), end.isoformat())
            data[v] = np.clip(arr, *bc.PHYS[v])
            if dates is None:
                dates = dd
            print(f"   {v}: {data[v].shape}  (bulk annual file)")
            continue
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
    # HDWI reuses the weather already downloaded above -- no extra network.
    render_hdwi(data, dates[-1], shape2d)


# HDWI nested bands. A cell is painted by the HIGHEST threshold it exceeds,
# and anything under the 90th percentile is left transparent -- that is ~90%
# of the map on a normal day, and painting it would bury the terrain while
# saying nothing. The four bands are not evenly spaced in frequency: p90 is
# 1 day in 10, p95 is 1 in 20, p97 is 1 in 33, p99 is 1 in 100 (three or four
# days a year at a given cell). So the map should empty out fast as you climb
# -- a mostly-blank p99 is the correct answer, not a broken one.
# Each cell is painted by the HIGHEST threshold it exceeds, so these are
# EXCLUSIVE ranges, not cumulative ones. Naming them "> 90th" and scoring
# them against "10% of days" was wrong on both counts: the yellow band is
# 90th-to-95th, which is 5% of days, and the counts do not have to decrease
# monotonically. Third value is the label, fourth is the share of days an
# average cell spends in that band.
HDWI_BANDS = [
    (90, (250, 204,  60, 150), '90-95th', 0.05),
    (95, (245, 130,  30, 180), '95-97th', 0.02),
    (97, (215,  45,  35, 210), '97-99th', 0.02),
    (99, (255, 255, 255, 240), '99th+',   0.01),
]


def hdwi_from_grid(tmmx_k, rmin_pct, vs_ms):
    """Same formula as build_hdwi_climatology and index.html.

    Three copies of this expression now exist -- Python for the climatology,
    Python here for the daily value, JavaScript in the page. They must agree
    exactly or the percentiles are meaningless, which is why the constants
    are written out identically in all three rather than refactored into
    something clever.
    """
    t_c = tmmx_k - 273.15
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    vpd = np.maximum(0.0, (1.0 - rmin_pct / 100.0) * es)
    return np.maximum(0.0, vs_ms * vpd)


def _hdwi_lookup():
    for cand in (os.path.join(bc.WORK, "breakpoints", "hdwi_lookup.npz"),
                 "hdwi_lookup.npz", os.path.join("data", "hdwi_lookup.npz")):
        if os.path.exists(cand):
            return np.load(cand)
    return None


def render_hdwi(data, date, shape2d):
    lut = _hdwi_lookup()
    if lut is None:
        print("  (no hdwi_lookup.npz — skipping HDWI)")
        return
    curves, pcts = lut["curves"], list(lut["pcts"])

    h = hdwi_from_grid(
        np.nan_to_num(data["tmmx"][-1].ravel(), nan=288.0),
        np.nan_to_num(data["rmin"][-1].ravel(), nan=50.0),
        np.nan_to_num(data["vs"][-1].ravel(), nan=0.0))

    cls = np.zeros(h.size, dtype=np.int8)
    for i, (p, _c, _n, _e) in enumerate(HDWI_BANDS, start=1):
        thresh = curves[:, pcts.index(float(p))]
        # A cell with a zero threshold has no real climatology (ocean, or a
        # cell that is calm every single day). Never band those -- otherwise
        # any breeze at all reads as "above the 99th percentile".
        cls = np.where((h > thresh) & (thresh > 0.01), i, cls)

    from PIL import Image
    cc = np.load(os.path.join(bc.META, "climate_class.npy"))
    land = np.isfinite(cc).ravel()
    rgba = np.zeros((h.size, 4), dtype=np.uint8)
    for i, (_p, col, _n, _e) in enumerate(HDWI_BANDS, start=1):
        rgba[(cls == i) & land] = col
    Image.fromarray(rgba.reshape(shape2d + (4,)), "RGBA").save(
        os.path.join(OUT_DIR, "hdwi_latest.png"), optimize=True)

    lats = np.load(os.path.join(bc.META, "lat.npy"))
    lons = np.load(os.path.join(bc.META, "lons.npy"))
    nland = int(land.sum())
    counts = {n: int(((cls == i+1) & land).sum())
              for i, (_p, _c, n, _e) in enumerate(HDWI_BANDS)}
    # Cumulative share above the 90th is the headline number -- it is what
    # says whether today is unusual across the region as a whole.
    above90 = int(((cls >= 1) & land).sum())
    with open(os.path.join(OUT_DIR, "hdwi_latest.json"), "w") as f:
        json.dump({
            "valid_date": date.isoformat(),
            "generated_utc": dt.datetime.now(dt.timezone.utc)
                               .isoformat().replace("+00:00", "Z"),
            "bounds": [[float(lats.min()), float(lons.min())],
                       [float(lats.max()), float(lons.max())]],
            "bands": [n for _p, _c, n, _e in HDWI_BANDS],
            "colors": ["#%02X%02X%02X" % c[:3] for _p, c, _n, _e in HDWI_BANDS],
            "above_p90_cells": above90,
            "above_p90_fraction": round(above90 / nland, 5),
            "band_cell_counts": counts,
            "band_fractions": {k: round(v / nland, 5) for k, v in counts.items()},
            "climatology": "gridMET 1979-2017",
            "source": "HDWI = wind x VPD, gridMET daily max temp / min RH / "
                      "mean wind; percentiles vs. per-cell 1979-2017",
        }, f, indent=2)

    print(f"\nHDWI valid {date}")
    for i, (_p, _c, n, expect) in enumerate(HDWI_BANDS, start=1):
        frac = counts[n] / nland
        print(f"   {n:>8}: {counts[n]:>8,} cells {100*frac:>5.2f}%  "
              f"(average day {100*expect:>4.0f}%)")
    print(f"   {'above p90':>8}: {above90:>8,} cells "
          f"{100*above90/nland:>5.2f}%  (average day   10%)")
    # These are SPATIAL shares on one day, not per-cell frequencies over
    # time, so an individual band can exceed its long-run share without
    # anything being wrong. The cumulative line is the one to read.


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
