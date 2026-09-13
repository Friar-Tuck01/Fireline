"""
HDWI climatology from HRRR, so the percentile can show TODAY.

WHY THIS EXISTS
    The gridMET HDWI climatology is excellent -- 39 years, 14,245 days per
    cell -- but gridMET is observational and runs 2-4 days behind. Feeding
    HRRR into it was measured and rejected: even after correcting the
    definitional mismatch, only 30% of values landed within 5 percentile
    points and correlation was 0.748. That scatter is not a fixable offset
    (see compare_hrrr_gridmet.py).

    The percentile only means something when the climatology and today's
    value come from the same source. So: build the climatology from HRRR
    too. Fewer years, but internally consistent, and current.

THE DEFINITION IS A SINGLE AFTERNOON HOUR, DELIBERATELY
    Reproducing gridMET's daily max-T / min-RH / mean-wind from HRRR needs
    all 24 hours: ~385,000 range requests and 578 GB. Sampling one hour
    needs 16,000 and ~24 GB.

    But this is not merely a concession to cost. We are no longer imitating
    gridMET, so we get to choose the definition, and a real 21Z snapshot is
    arguably better than a proxy that pairs the day's hottest hour with its
    driest and its average wind -- three moments that never co-occurred.
    21Z is 2pm Pacific, 3pm Mountain: near peak burning across the domain.

KNOWN LIMITATIONS, both stated in the layer
    * ~11 years, not 39. The 97th percentile rests on roughly 120 days per
      cell rather than 427. Usable, but noisier in the tail, which is
      exactly where fire people look.
    * HRRR changed version several times over the archive (v2 in 2016, v3 in
      2018, v4 in 2020). A climatology spanning those is not perfectly
      homogeneous. Reanalyses avoid this by rerunning one model version over
      all years; an operational archive cannot.

    python3 build_hdwi_hrrr_climatology.py run        # ~7 h, resumable
    python3 build_hdwi_hrrr_climatology.py quantiles
    python3 build_hdwi_hrrr_climatology.py status
"""

import datetime as dt
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np

import build_climatology as bc

AWS_HRRR = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# SAMPLING THE DIURNAL CYCLE, NOT JUST THE AFTERNOON
#
# The first version sampled 21Z only -- 2pm Pacific, near peak burning. That
# was wrong for this domain, and not by a little. The West's most dangerous
# fire weather is frequently NOCTURNAL: Diablo, Santa Ana and Mono offshore
# wind events typically peak overnight into early morning, and they are the
# patterns behind Tubbs, Camp and Thomas.
#
# An afternoon-only climatology would not merely under-sample those events,
# it would be blind to them. A cell could see its worst HDWI of the decade at
# 3am and the climatology would never record it -- so a future 3am event
# would be ranked against a distribution built entirely from ordinary
# afternoons, and would read as far less unusual than it is. The layer would
# fail exactly when it matters most.
#
# Four samples a day at 00/06/12/18Z are, in Pacific summer, roughly 5pm,
# 11pm, 5am and 11am: late afternoon, night, pre-dawn and late morning. We
# take the daily maximum across them.
#
# This is a SAMPLED maximum, not a true one -- a 3-hour offshore surge
# peaking at 02Z can still slip between samples. But it samples the diurnal
# cycle instead of assuming fire weather is an afternoon phenomenon, and that
# assumption is what actually breaks.
SAMPLE_HOURS = (0, 6, 12, 18)
MAX_WORKERS = 10          # concurrent range requests; network-bound
BATCH_DAYS = 10           # bounds peak memory on raw GRIB bytes
START = dt.date(2015, 1, 1)
OUT = os.path.join(bc.WORK, "hdwi_hrrr")
SCALE = 10.0

FIELDS = {
    "t2m": ":TMP:2 m above ground:",
    "d2m": ":DPT:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
}

PCTS = np.array(
    list(range(0, 90, 5)) + list(range(90, 100)) + [99.5, 100], dtype=np.float64)


def _get(url, rng=None, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "fireline/1.0"})
    if rng:
        req.add_header("Range", f"bytes={rng}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _url(day, hour):
    return (f"{AWS_HRRR}/hrrr.{day:%Y%m%d}/conus/"
            f"hrrr.t{hour:02d}z.wrfsfcf00.grib2")


def _ranges(day, hour):
    """Byte ranges for our four fields, from the .idx sidecar."""
    base = _url(day, hour)
    idx = _get(base + ".idx", timeout=60).decode().splitlines()
    starts = [int(l.split(":")[1]) for l in idx]
    out = {}
    for key, pat in FIELDS.items():
        line = next((i for i, l in enumerate(idx) if pat in l), None)
        if line is None:
            raise RuntimeError(f"{pat} missing")
        a = starts[line]
        b = starts[line + 1] - 1 if line + 1 < len(starts) else ""
        out[key] = f"{a}-{b}"
    return base, out


def _hdwi_from_blobs(blobs):
    """Parse four GRIB messages and return the HDWI field.

    Parsing happens on ONE thread. eccodes (under pygrib) is not reliably
    thread-safe, and the bottleneck here is the network anyway -- so the
    fetches are concurrent and the decoding is not.
    """
    import pygrib
    vals, latlon = {}, None
    for key, blob in blobs.items():
        m = pygrib.fromstring(blob)
        vals[key] = m.values.astype(np.float32)
        if latlon is None:
            la, lo = m.latlons()
            latlon = (la.astype(np.float32), lo.astype(np.float32))
    t = vals["t2m"] - 273.15
    td = vals["d2m"] - 273.15
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    # VPD from dewpoint is algebraically identical to (1-RH/100)*es, so this
    # is the same quantity the gridMET climatology used -- only the source
    # and the sampling differ.
    vpd = np.maximum(0.0, es - e)
    return np.maximum(0.0, np.hypot(vals["u10"], vals["v10"]) * vpd), latlon


def hrrr_hdwi(day, nn, hour=None):
    """HDWI on the gridMET grid for one analysis hour."""
    hour = SAMPLE_HOURS[-1] if hour is None else hour
    base, rngs = _ranges(day, hour)
    blobs = {k: _get(base, rng=r) for k, r in rngs.items()}
    h, latlon = _hdwi_from_blobs(blobs)
    return h.ravel()[nn] if nn is not None else (h, latlon)


def _fetch_day_hours(day):
    """All SAMPLE_HOURS for one day, fetched concurrently.

    Returns {hour: {field: bytes}}; hours that fail are simply absent.
    """
    from concurrent.futures import ThreadPoolExecutor
    tasks = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_ranges, day, h): h for h in SAMPLE_HOURS}
        plans = {}
        for f, h in futs.items():
            try:
                plans[h] = f.result()
            except Exception:
                pass
        jobs = {}
        for h, (base, rngs) in plans.items():
            for k, r in rngs.items():
                jobs[ex.submit(_get, base, r)] = (h, k)
        for f, (h, k) in jobs.items():
            try:
                tasks.setdefault(h, {})[k] = f.result()
            except Exception:
                tasks.pop(h, None)
    return {h: b for h, b in tasks.items() if len(b) == len(FIELDS)}


def _build_nn():
    """Nearest HRRR cell for every gridMET cell. Fixed grids, so cache it."""
    cache = os.path.join(bc.WORK, "hrrr_to_gridmet_nn.npy")
    if os.path.exists(cache):
        return np.load(cache)
    print("building HRRR -> gridMET index (once)")
    h, (hlat, hlon) = hrrr_hdwi(dt.date.today() - dt.timedelta(days=3), None)
    glat2 = np.load(os.path.join(bc.META, "lat.npy"))
    glon = np.load(os.path.join(bc.META, "lons.npy"))
    glon2 = np.tile(glon, (glat2.shape[0], 1))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([hlat.ravel(), hlon.ravel()]))
    _, nn = tree.query(np.column_stack([glat2.ravel(), glon2.ravel()]),
                       workers=-1)
    nn = nn.astype(np.int32)
    np.save(cache, nn)
    print(f"  {nn.size:,} gridMET cells matched")
    return nn


def run():
    os.makedirs(OUT, exist_ok=True)
    nn = _build_nn()
    end = dt.date.today() - dt.timedelta(days=2)
    years = range(START.year, end.year + 1)

    for yr in years:
        path = os.path.join(OUT, f"hdwi_{yr}.npy")
        marker = path + ".days"
        if os.path.exists(path):
            print(f"  {yr} already done")
            continue
        d0 = max(START, dt.date(yr, 1, 1))
        d1 = min(end, dt.date(yr, 12, 31))
        days, rows, nhours = [], [], []
        t0 = time.time()
        d = d0
        while d <= d1:
            try:
                blobs = _fetch_day_hours(d)
            except Exception as e:
                print(f"     {d} skipped ({str(e)[:60]})")
                d += dt.timedelta(days=1)
                continue
            best = None
            for h in sorted(blobs):
                try:
                    field, _ = _hdwi_from_blobs(blobs[h])
                except Exception:
                    continue
                v = field.ravel()[nn]
                best = v if best is None else np.maximum(best, v)
            if best is None:
                # HRRR has real gaps -- outages, missing cycles, the version
                # transitions. A missing day is a smaller problem than a
                # halted multi-hour run, so record and continue.
                print(f"     {d} no usable hours")
            else:
                rows.append(np.rint(best * SCALE).clip(0, 65535)
                            .astype(np.uint16))
                days.append(d.isoformat())
                nhours.append(len(blobs))
            d += dt.timedelta(days=1)
            if len(days) % 25 == 0 and days:
                el = time.time() - t0
                rate = el / max(len(days), 1)
                print(f"     {yr}: {len(days)} days, {el/60:.0f} min, "
                      f"{rate:.1f} s/day, mean {np.mean(nhours):.1f} hours/day")
        if not rows:
            print(f"  {yr}: nothing retrieved")
            continue
        np.save(path, np.array(rows, dtype=np.uint16))
        with open(marker, "w") as f:
            f.write("\n".join(days))
        print(f"  {yr}: {len(days)} days in {(time.time()-t0)/60:.0f} min "
              f"(mean {np.mean(nhours):.1f} of {len(SAMPLE_HOURS)} hours "
              f"available)")
    print("\nall years done — now run `quantiles`")


def quantiles():
    files = sorted(f for f in os.listdir(OUT)
                   if f.startswith("hdwi_") and f.endswith(".npy"))
    if not files:
        sys.exit("no year files — run `run` first")
    mm = [np.load(os.path.join(OUT, f), mmap_mode="r") for f in files]
    ndays = sum(a.shape[0] for a in mm)
    cc = np.load(os.path.join(bc.META, "climate_class.npy"))
    ncell = cc.size
    land = np.isfinite(cc).ravel()
    print(f"{len(files)} years, {ndays:,} days, {ncell:,} cells")
    if ndays < 2000:
        print(f"  WARNING: {ndays} days is thin for a 99th percentile")

    curves = np.zeros((ncell, len(PCTS)), dtype=np.float32)
    TILE = 5000
    for s in range(0, ncell, TILE):
        e = min(s + TILE, ncell)
        blk = np.concatenate([a[:, s:e] for a in mm], axis=0).astype(np.float32)
        curves[s:e] = (np.percentile(blk, PCTS, axis=0).T / SCALE).astype(np.float32)
        if s % 50000 == 0:
            i90 = list(PCTS).index(90)
            print(f"  cells {s:>7,}-{e:>7,}  mean p90 {curves[s:e,i90].mean():6.1f}")

    path = os.path.join(bc.WORK, "breakpoints", "hdwi_hrrr_lookup.npz")
    np.savez_compressed(path, curves=curves, pcts=PCTS.astype(np.float32),
                        shape=np.array(cc.shape), ndays=np.array([ndays]),
                        hours_utc=np.array(SAMPLE_HOURS),
                        aggregation=np.array(["daily max over sampled hours"]),
                        years=np.array([files[0][5:9], files[-1][5:9]]))
    print(f"\nwrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    print("\nverifying against the definition")
    idx = np.where(land)[0][::4000][:40]
    o90, o95 = [], []
    for c in idx:
        ser = np.concatenate([a[:, c] for a in mm]).astype(np.float32) / SCALE
        o90.append(np.mean(ser > curves[c, list(PCTS).index(90)]))
        o95.append(np.mean(ser > curves[c, list(PCTS).index(95)]))
    print(f"  above p90: {100*np.mean(o90):.1f}%  (want 10%)")
    print(f"  above p95: {100*np.mean(o95):.1f}%  (want  5%)")


def status():
    n = len(os.listdir(OUT)) // 2 if os.path.isdir(OUT) else 0
    end = dt.date.today() - dt.timedelta(days=2)
    print(f"years complete: {n}/{end.year - START.year + 1}")
    if os.path.isdir(OUT):
        gb = sum(os.path.getsize(os.path.join(OUT, f))
                 for f in os.listdir(OUT)) / 1e9
        print(f"  {gb:.2f} GB")
    p = os.path.join(bc.WORK, "breakpoints", "hdwi_hrrr_lookup.npz")
    print(f"lookup: {'yes' if os.path.exists(p) else 'not built'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"run": run, "quantiles": quantiles, "status": status}[cmd]()
