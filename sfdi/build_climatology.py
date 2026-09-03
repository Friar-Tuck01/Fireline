"""
Build the SFDI climatology: run our NFDRS engine over gridMET 1979-2017.

This is the expensive, one-time step. It downloads gridMET meteorology a year
at a time, runs the validated engine forward across the western US grid, and
writes ERC and BI as compact int16 arrays. Raw downloads are deleted as soon
as a year is processed, so peak disk stays modest even though the cumulative
download is large.

    python3 build_climatology.py prepass   # climate-class map (needs pr only)
    python3 build_climatology.py run       # the long one
    python3 build_climatology.py status    # what is done so far

Wrap it in caffeinate so the Mac does not sleep mid-run:

    caffeinate -i python3 build_climatology.py run

RESUMABILITY. Sleep drops in-flight connections and the process would
otherwise die. Three defenses:
  * each year's output is written and marked complete before moving on, so
    finished years are never redone
  * downloads retry with backoff instead of dying on one dropped socket
  * files download to a .tmp name and are renamed only on success. A
    half-written file that LOOKS complete is the dangerous failure -- it
    would feed truncated weather into the engine and produce plausible
    garbage, exactly the class of bug that cost us three rounds already.

STATE CARRIES ACROSS YEARS. ERC is a build-up index; 1000-hour fuels have
weeks of memory. The engine state is pickled at each year boundary and
reloaded, so the series is continuous rather than 39 independent years.
"""

import os
import pickle
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import nfdrs78 as n

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE = "http://www.northwestknowledge.net/metdata/data"

# The paper's climatology window (Jolly et al. 2019). Keeping this rather
# than a modern normal means our SFDI *is* the published SFDI and their
# validation statistics describe our layer without an asterisk.
YEAR_START, YEAR_END = 1979, 2017

# Western US. gridMET is 1/24 degree on a 585x1386 CONUS grid spanning
# 25.05-49.42 N, -124.79 to -67.04 E.
WEST, EAST = -125.0, -102.0
SOUTH, NORTH = 31.0, 49.5

MET_VARS = ["tmmx", "tmmn", "rmax", "rmin", "vs", "pr", "srad"]
VARNAME = {
    "tmmx": "daily_maximum_temperature",
    "tmmn": "daily_minimum_temperature",
    "rmax": "daily_maximum_relative_humidity",
    "rmin": "daily_minimum_relative_humidity",
    "vs": "daily_mean_wind_speed",
    "pr": "precipitation_amount",
    "srad": "daily_mean_shortwave_radiation_at_surface",
}

# Physical bounds, checked every year. If gridMET's packing or units ever
# change, this fails loudly instead of silently poisoning the run.
# Corruption detectors, NOT plausibility filters. The job is to catch a
# packing/units change (which shifts EVERY value), not to police rare weather.
# Set them wide enough that genuine extremes pass and only nonsense trips them.
#
# pr was originally capped at 500 mm and fired on 2005 with a 690 mm day --
# ~27 inches, near California's 24-hour record. That is a real atmospheric
# river hitting a high-elevation coastal cell in a 4 km interpolated product,
# not corruption: the minimum was still 0.0, and 24 prior years passed the
# same check. Raised to 1200 mm, below the world record of ~1825 mm.
PHYS = {
    "tmmx": (220.0, 340.0), "tmmn": (210.0, 330.0),
    "rmax": (0.0, 100.0), "rmin": (0.0, 100.0),
    "vs": (0.0, 60.0), "pr": (0.0, 1200.0), "srad": (0.0, 600.0),
}

# A packing change corrupts the whole field; a weather extreme is a handful of
# cells. If more than this fraction is out of range, it is not weather.
BAD_FRACTION_LIMIT = 1e-4

WORK = os.environ.get("SFDI_WORK", os.path.expanduser("~/fireline-sfdi-work"))
RAW = os.path.join(WORK, "raw")
OUT = os.path.join(WORK, "erc_bi")
META = os.path.join(WORK, "meta")

MONTH_CHUNK = True   # read a month at a time to keep memory near 250 MB


def _paths():
    for d in (WORK, RAW, OUT, META):
        os.makedirs(d, exist_ok=True)


def _require_netcdf4():
    """netCDF4 only -- deliberately NOT xarray.

    xarray probes for dask on every array operation. A dask install whose
    __version__ is the string 'unknown' makes that probe raise
    InvalidVersion, which has nothing to do with our data and is painful to
    debug. netCDF4 is a thinner layer, applies scale_factor/add_offset
    automatically, and has no such dependency.
    """
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        sys.exit("Needs netCDF4:\n"
                 "    conda install -y netcdf4\n"
                 "  (or: pip3 install netcdf4)")


def _subset_index(ds):
    """Index ranges for our western-US window.

    gridMET's lat runs north-to-south, but we search rather than assume it --
    if the ordering ever flips, argwhere still finds the right rows instead
    of silently returning an empty slice.
    """
    lats = ds.variables["lat"][:]
    lons = ds.variables["lon"][:]
    la = np.argwhere((lats >= SOUTH) & (lats <= NORTH)).ravel()
    lo = np.argwhere((lons >= WEST) & (lons <= EAST)).ravel()
    if la.size == 0 or lo.size == 0:
        raise ValueError(
            f"empty subset: lat {lats.min():.2f}..{lats.max():.2f}, "
            f"lon {lons.min():.2f}..{lons.max():.2f} vs window "
            f"{SOUTH}..{NORTH}, {WEST}..{EAST}")
    return (slice(la[0], la[-1] + 1), slice(lo[0], lo[-1] + 1),
            np.asarray(lats[la[0]:la[-1] + 1]),
            np.asarray(lons[lo[0]:lo[-1] + 1]))


def _data_var(ds, path=""):
    """Find the data variable by shape instead of by name.

    The bulk annual files at northwestknowledge.net use CF standard names
    (air_temperature, relative_humidity, wind_speed, ...) while the THREDDS
    aggregations use descriptive ones (daily_maximum_temperature, ...). Two
    variables, two naming schemes, and `pr` happens to be spelled the same in
    both -- which is why the prepass worked and everything else did not.

    Rather than hardcode a third set of guessed names, take the only variable
    with three dimensions. Coordinates are 1-D, so this is unambiguous, and
    it keeps working if either naming scheme changes again.
    """
    cands = [k for k, var in ds.variables.items() if var.ndim == 3]
    if len(cands) != 1:
        raise ValueError(
            f"expected exactly one 3-D variable in {path or 'file'}, "
            f"found {cands}. Variables present: {list(ds.variables)}")
    return cands[0]


def _read(ds, varname, tslice, ilat, ilon):
    """Read a time chunk as float32 with masked values as NaN."""
    v = ds.variables[varname]
    arr = v[tslice, ilat, ilon]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=np.float32)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download(var, year, tries=6):
    """Fetch one gridMET annual file. Atomic and resumable."""
    dest = os.path.join(RAW, f"{var}_{year}.nc")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    url = f"{BASE}/{var}_{year}.nc"
    tmp = dest + ".tmp"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "fireline-sfdi/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(tmp, "wb") as f:
                nbytes = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    nbytes += len(chunk)
            if nbytes < 1_000_000:
                raise IOError(f"suspiciously small ({nbytes} bytes)")
            os.replace(tmp, dest)      # atomic: only now does it look done
            return dest
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            wait = min(60, 2 ** attempt)
            print(f"      retry {attempt+1}/{tries} for {var}_{year} "
                  f"({type(e).__name__}: {e}); waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up on {var}_{year}")


# --------------------------------------------------------------------------
# Climate class prepass
# --------------------------------------------------------------------------
def prepass():
    """Derive an NFDRS climate class map from mean annual precipitation.

    NFDRS climate class (1 arid ... 4 humid) sets greenup length, dormant
    woody moisture, the herbaceous regressions and the assumed rainfall rate.
    Operationally a human assigns it per station. On a grid we have to infer
    it, and the precipitation thresholds below are a MODELING CHOICE, not
    something GTR-82 specifies -- the same category as our greenup rule and
    the state-of-weather binning. Flagged here so it is a known knob.

    Uses 10 years spread across the record rather than all 39: mean annual
    precipitation is stable, and this keeps the prepass to ~10 downloads.
    """
    _paths(); _require_netcdf4()
    import netCDF4

    out = os.path.join(META, "climate_class.npy")
    if os.path.exists(out):
        print(f"already have {out}")
        return

    years = list(range(1981, 2018, 4))[:10]
    print(f"Climate-class prepass using pr for {years}")
    total = None
    valid = None
    for y in years:
        p = download("pr", y)
        with netCDF4.Dataset(p) as ds:
            ilat, ilon, lats, lons = _subset_index(ds)
            vname = _data_var(ds, p)
            ndays = ds.variables[vname].shape[0]
            annual = None
            # Accumulate in month-sized chunks: a full year of the subset is
            # ~360 MB as float32 and there is no reason to hold it all.
            for a in range(0, ndays, 31):
                b = min(a + 31, ndays)
                chunk = _read(ds, vname, slice(a, b), ilat, ilon)
                if valid is None:
                    # Land mask from the DATA, not from the accumulated sum.
                    # np.nansum returns 0.0 for an all-NaN slice, so ocean
                    # cells would otherwise come through as "0 mm annual
                    # precipitation", pass the isfinite check, and get
                    # classified as arid instead of masked out.
                    valid = np.isfinite(chunk[0])
                part = np.nansum(chunk, axis=0)
                annual = part if annual is None else annual + part
        total = annual if total is None else total + annual
        print(f"  {y}: grid mean annual precip "
              f"{np.nanmean(annual):.0f} mm  ({annual.shape})")
        os.remove(p)

    mm = total / len(years)
    inches = mm / 25.4
    # <10 in arid, 10-20 semi-arid, 20-40 sub-humid, >40 humid
    cc = np.select([inches < 10, inches < 20, inches < 40],
                   [1.0, 2.0, 3.0], default=4.0).astype(np.float32)
    cc[~valid] = np.nan
    mm = np.where(valid, mm, np.nan)
    print(f"  land cells: {int(valid.sum()):,} of {valid.size:,} "
          f"({100*valid.mean():.1f}%)")
    np.save(out, cc)
    np.save(os.path.join(META, "annual_precip_mm.npy"), mm)
    for k in (1, 2, 3, 4):
        print(f"  class {k}: {100*np.nanmean(cc == k):.1f}% of cells")
    print(f"wrote {out}  shape {cc.shape}")


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------
def _load_year(year):
    """Yield (slice_index, dict-of-arrays) chunks for one year."""
    import netCDF4
    handles, vnames = {}, {}
    for v in MET_VARS:
        path = download(v, year)
        handles[v] = netCDF4.Dataset(path)
        vnames[v] = _data_var(handles[v], path)
    if year == YEAR_START:
        print("    detected variables: "
              + ", ".join(f"{k}->{vnames[k]}" for k in MET_VARS))
    ilat, ilon, _, _ = _subset_index(handles["tmmx"])
    ndays = handles["tmmx"].variables[vnames["tmmx"]].shape[0]
    step = 31 if MONTH_CHUNK else ndays
    try:
        for a in range(0, ndays, step):
            b = min(a + step, ndays)
            chunk = {}
            for v in MET_VARS:
                arr = _read(handles[v], vnames[v], slice(a, b), ilat, ilon)
                lo, hi = PHYS[v]
                fin = np.isfinite(arr)
                if fin.any():
                    bad = fin & ((arr < lo) | (arr > hi))
                    nbad = int(bad.sum())
                    if nbad:
                        frac = nbad / int(fin.sum())
                        mn, mx = np.nanmin(arr[fin]), np.nanmax(arr[fin])
                        # Report the count, not just the extreme value. One
                        # cell out of millions is weather; a large fraction
                        # means the units or packing moved under us.
                        msg = (f"{v} {year} days {a}-{b}: {nbad:,} values "
                               f"({100*frac:.4f}%) outside [{lo}, {hi}]; "
                               f"field range {mn:.1f}-{mx:.1f}")
                        if frac > BAD_FRACTION_LIMIT:
                            raise ValueError(
                                msg + ". That is too much of the field to be "
                                "weather -- gridMET packing or units likely "
                                "changed. Stopping.")
                        print(f"      NOTE {msg} -- isolated, treated as a "
                              f"genuine extreme and clipped")
                        arr = np.clip(arr, lo, hi)
                chunk[v] = arr
            yield a, b, chunk
    finally:
        for ds in handles.values():
            ds.close()


def run():
    _paths(); _require_netcdf4()
    cc_path = os.path.join(META, "climate_class.npy")
    if not os.path.exists(cc_path):
        sys.exit("Run `prepass` first to build the climate-class map.")
    cc2d = np.load(cc_path)
    shape2d = cc2d.shape
    land = np.isfinite(cc2d)
    climat = np.where(land, cc2d, 2.0).astype(np.float64).ravel()
    ncell = climat.size
    print(f"grid {shape2d} = {ncell:,} cells, {100*land.mean():.0f}% land")

    lat2d = np.load(os.path.join(META, "lat.npy")) \
        if os.path.exists(os.path.join(META, "lat.npy")) else None
    if lat2d is None:
        import netCDF4
        p = download("tmmx", YEAR_START)
        with netCDF4.Dataset(p) as ds:
            _, _, lats, lons = _subset_index(ds)
        lat2d = np.repeat(lats[:, None], len(lons), axis=1).astype(np.float32)
        np.save(os.path.join(META, "lat.npy"), lat2d)
        np.save(os.path.join(META, "lons.npy"), lons)
    lat_flat = lat2d.ravel().astype(np.float64)

    state_path = os.path.join(META, "state.pkl")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st, next_year = pickle.load(f)
        print(f"resuming at {next_year}")
    else:
        st = n.NFDRSState(ncell, climate_class=climat)
        next_year = YEAR_START
        # Spin-up: 1000-hour fuels start from a crude guess and need time to
        # forget it. Run the first year twice and discard the first pass.
        print(f"spin-up pass over {YEAR_START} (discarded)")
        _process_year(st, YEAR_START, lat_flat, shape2d, write=False)

    for year in range(next_year, YEAR_END + 1):
        t0 = time.time()
        _process_year(st, year, lat_flat, shape2d, write=True)
        for v in MET_VARS:
            p = os.path.join(RAW, f"{v}_{year}.nc")
            if os.path.exists(p):
                os.remove(p)
        with open(state_path, "wb") as f:
            pickle.dump((st, year + 1), f)
        print(f"  {year} done in {(time.time()-t0)/60:.1f} min "
              f"(raw deleted, state saved)")
    print("\nclimatology complete")


def _process_year(st, year, lat_flat, shape2d, write=True):
    erc_days, bi_days = [], []
    print(f"[{year}]")
    for a, b, chunk in _load_year(year):
        for i in range(b - a):
            jday = a + i + 1
            srad = np.nan_to_num(chunk["srad"][i].ravel(), nan=200.0)
            clear = _clear_sky(lat_flat, jday)
            sow = n.state_of_weather_from_srad(srad, clear)
            out = n.step(st, dict(
                tmax_f=_kf(chunk["tmmx"][i]), tmin_f=_kf(chunk["tmmn"][i]),
                rhmax=np.nan_to_num(chunk["rmax"][i].ravel(), nan=50.0),
                rhmin=np.nan_to_num(chunk["rmin"][i].ravel(), nan=25.0),
                tobs_f=_kf(chunk["tmmx"][i]),
                rhobs=np.nan_to_num(chunk["rmin"][i].ravel(), nan=25.0),
                sow=sow,
                ppt_in=np.nan_to_num(chunk["pr"][i].ravel(), nan=0.0) / 25.4,
                ws_mph=np.nan_to_num(chunk["vs"][i].ravel(), nan=4.0)
                * 0.914 * 2.23694,
                lat=lat_flat, jday=jday))
            if write:
                erc_days.append(out["erc"].astype(np.int16))
                bi_days.append(out["bi"].astype(np.int16))
        print(f"    days {a+1}-{b}")
    if write:
        np.save(os.path.join(OUT, f"erc_{year}.npy"),
                np.array(erc_days, dtype=np.int16))
        np.save(os.path.join(OUT, f"bi_{year}.npy"),
                np.array(bi_days, dtype=np.int16))


def _kf(arr):
    return np.nan_to_num(arr.ravel(), nan=288.0) * 9 / 5 - 459.67


def _clear_sky(lat_deg, jday):
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * jday / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * jday / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0)
    ws = np.arccos(x)
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.75 * ra * 11.574


def status():
    _paths()
    done = sorted(int(f[4:8]) for f in os.listdir(OUT) if f.startswith("erc_"))
    print(f"work dir: {WORK}")
    print(f"years complete: {len(done)}/{YEAR_END-YEAR_START+1}")
    if done:
        print(f"  {done[0]}-{done[-1]}")
    tot = sum(os.path.getsize(os.path.join(OUT, f))
              for f in os.listdir(OUT)) / 1e9
    raw = sum(os.path.getsize(os.path.join(RAW, f))
              for f in os.listdir(RAW)) / 1e9
    print(f"output: {tot:.1f} GB   raw pending: {raw:.1f} GB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"prepass": prepass, "run": run, "status": status}[cmd]()
