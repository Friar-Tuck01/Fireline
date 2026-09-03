"""
Validate nfdrs78 against gridMET's own published ERC and BI.

This is the check that matters. Everything in test_nfdrs78.py only proves the
engine is self-consistent -- it cannot tell us whether our ERC means the same
thing as the ERC used to build the climatology. Since SFDI is a percentile,
a constant offset between the two would be invisible in the output and would
silently mislabel every day on the map.

Strategy: pull raw gridMET meteorology for a handful of points spread across
western climate types, run our engine on it, and diff against the erc/bi that
gridMET publishes for those same cells and days.

Run in three stages:

    python3 validate_gridmet.py probe      # discover dataset paths + var names
    python3 validate_gridmet.py fetch      # download point series to work/
    python3 validate_gridmet.py compare    # run engine, report the diff

NOTE ON THE PROBE STEP: the exact gridMET THREDDS dataset paths and internal
variable names below are a best guess and are almost certainly not all
correct. `probe` asks the server what it actually has and prints it. Fix
GRIDMET_VARS from that output before running `fetch`. This is deliberate --
hardcoding unverified endpoint strings is how you end up debugging a 404 at
midnight thinking your science is wrong.
"""

import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import nfdrs78 as n

THREDDS = "http://thredds.northwestknowledge.net:8080/thredds"
WORK = "work"

# variable key -> (dataset filename, expected internal variable name)
# VERIFY THESE WITH `probe` BEFORE TRUSTING THEM.
GRIDMET_VARS = {
    "tmmx": ("agg_met_tmmx_1979_CurrentYear_CONUS.nc", "daily_maximum_temperature"),
    "tmmn": ("agg_met_tmmn_1979_CurrentYear_CONUS.nc", "daily_minimum_temperature"),
    "rmax": ("agg_met_rmax_1979_CurrentYear_CONUS.nc", "daily_maximum_relative_humidity"),
    "rmin": ("agg_met_rmin_1979_CurrentYear_CONUS.nc", "daily_minimum_relative_humidity"),
    "vs":   ("agg_met_vs_1979_CurrentYear_CONUS.nc", "daily_mean_wind_speed"),
    "pr":   ("agg_met_pr_1979_CurrentYear_CONUS.nc", "precipitation_amount"),
    "srad": ("agg_met_srad_1979_CurrentYear_CONUS.nc",
             "daily_mean_shortwave_radiation_at_surface"),
    # Both of these needed the daily_mean_ prefix -- confirmed against the
    # server's own dataset.xml via `probe`, not guessed. The -g suffix
    # confirms gridMET publishes Fuel Model G, matching our engine.
    "erc":  ("agg_met_erc_1979_CurrentYear_CONUS.nc",
             "daily_mean_energy_release_component-g"),
    "bi":   ("agg_met_bi_1979_CurrentYear_CONUS.nc",
             "daily_mean_burning_index_g"),
}

# Spread across western climate types so a bias that only shows up in one
# regime cannot hide. NFDRS climate class: 1 arid, 2 semi-arid, 3 sub-humid,
# 4 humid. The class assignments are judgement calls and are themselves worth
# revisiting if a single site is the only one that disagrees.
SITES = [
    dict(name="reno_nv",      lat=39.53, lon=-119.81, climat=1),
    dict(name="boise_id",     lat=43.62, lon=-116.21, climat=2),
    dict(name="missoula_mt",  lat=46.87, lon=-113.99, climat=3),
    dict(name="redding_ca",   lat=40.59, lon=-122.39, climat=2),
    dict(name="flagstaff_az", lat=35.20, lon=-111.65, climat=2),
    dict(name="bend_or",      lat=44.06, lon=-121.31, climat=3),
]

# Long enough for the 1000-hour fuels to forget their crude initialization.
# The first SPINUP_DAYS are computed but excluded from the comparison.
FETCH_START = "2018-01-01"
FETCH_END = "2023-12-31"
SPINUP_DAYS = 730


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "fireline-sfdi/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def probe():
    """Ask the server what these datasets actually contain."""
    print("Probing gridMET THREDDS. For each dataset we fetch dataset.xml and\n"
          "list the grid variables it advertises.\n")
    for key, (fname, guessed) in GRIDMET_VARS.items():
        url = f"{THREDDS}/ncss/{fname}/dataset.xml"
        print(f"--- {key}  ({fname})")
        print(f"    {url}")
        try:
            xml = _get(url, timeout=60)
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code} -- dataset path is wrong. Browse\n"
                  f"    {THREDDS}/catalog/agg_met/catalog.html to find the real one.\n")
            continue
        except Exception as e:
            print(f"    FAILED: {e}\n")
            continue

        # Crude but dependency-free: pull name="..." out of <grid> elements.
        names = []
        for chunk in xml.split("<grid ")[1:]:
            if 'name="' in chunk:
                names.append(chunk.split('name="')[1].split('"')[0])
        print(f"    variables: {names}")
        print(f"    guessed:   {guessed}")
        print(f"    {'OK' if guessed in names else '>>> MISMATCH, update GRIDMET_VARS'}\n")
        time.sleep(1.0)   # be polite to a university research server


def fetch():
    """Download a CSV point series per site per variable."""
    os.makedirs(WORK, exist_ok=True)
    for site in SITES:
        for key, (fname, varname) in GRIDMET_VARS.items():
            out = os.path.join(WORK, f"{site['name']}_{key}.csv")
            if os.path.exists(out):
                print(f"  have {out}")
                continue
            url = (f"{THREDDS}/ncss/{fname}?var={varname}"
                   f"&latitude={site['lat']}&longitude={site['lon']}"
                   f"&time_start={FETCH_START}T00:00:00Z"
                   f"&time_end={FETCH_END}T00:00:00Z&accept=csv")
            print(f"  GET {site['name']} {key}")
            try:
                body = _get(url)
            except Exception as e:
                print(f"    FAILED: {e}\n    url was: {url}")
                return
            with open(out, "w") as f:
                f.write(body)
            # One request per second, sequential. This is a shared research
            # server; hammering it is how public data access gets revoked.
            time.sleep(1.0)
    print("fetch complete")


# gridMET NetCDF stores packed integers. THREDDS NCSS hands back the RAW
# packed values in CSV without applying scale_factor/add_offset, so a
# relative humidity of 67% arrives as 670 and a temperature of 293 K arrives
# as 731. Unpacked naively, that produced 856 F "temperatures", EMC values
# that clipped to zero, and impossible 0.2% fuel moisture -- with output that
# still looked like a plausible fire-danger series.
#
# Derived by solving the observed raw ranges against Reno's real climate and
# spot-checked against a known day (2018-07-01 -> 94.7 F, correct). ERC and
# BI come back unpacked already.
PACKING = {
    "tmmx": (0.1, 220.0), "tmmn": (0.1, 220.0),
    "rmax": (0.1, 0.0), "rmin": (0.1, 0.0),
    "vs":   (0.1, 0.0), "pr":  (0.1, 0.0), "srad": (0.1, 0.0),
    "erc":  (1.0, 0.0), "bi":  (1.0, 0.0),
}

# Physically valid range after unpacking. Checked on every read so a future
# change to gridMET's packing fails loudly instead of silently poisoning the
# fuel moisture. This is the guard that was missing the first time.
PHYS_RANGE = {
    "tmmx": (230.0, 335.0), "tmmn": (220.0, 320.0),
    "rmax": (0.0, 100.0), "rmin": (0.0, 100.0),
    "vs":   (0.0, 40.0), "pr": (0.0, 500.0), "srad": (0.0, 500.0),
    "erc":  (0.0, 200.0), "bi": (0.0, 300.0),
}


def _read_series(site_name, key):
    """Return (dates, values) from an NCSS CSV point response, unpacked."""
    path = os.path.join(WORK, f"{site_name}_{key}.csv")
    with open(path) as f:
        rows = list(csv.reader(io.StringIO(f.read())))
    header = rows[0]
    # NCSS CSV puts time first; the data column is the last numeric one.
    tcol = 0
    vcol = len(header) - 1
    dates, vals = [], []
    for r in rows[1:]:
        if len(r) <= vcol or not r[vcol].strip():
            continue
        try:
            vals.append(float(r[vcol]))
        except ValueError:
            continue
        dates.append(r[tcol].strip())

    scale, offset = PACKING[key]
    arr = np.array(vals) * scale + offset

    lo, hi = PHYS_RANGE[key]
    bad = np.sum((arr < lo) | (arr > hi))
    if bad:
        raise ValueError(
            f"{site_name}/{key}: {bad} of {len(arr)} values fall outside the "
            f"physical range [{lo}, {hi}] after unpacking "
            f"(got {np.nanmin(arr):.1f} to {np.nanmax(arr):.1f}). "
            f"gridMET's packing may have changed -- check PACKING.")
    return dates, arr


def _clear_sky_srad(lat_deg, jday):
    """Clear-sky shortwave (W/m2) via FAO-56 extraterrestrial radiation.

    Used only to turn gridMET's srad into NFDRS's 0-3 state-of-weather code.
    Rso = 0.75 * Ra is the standard approximation.
    """
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * jday / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * jday / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0)
    ws = np.arccos(x)
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.75 * ra * 11.574     # MJ/m2/day -> W/m2


def compare():
    print(f"{'site':<14}{'n':>6}{'bias':>9}{'MAE':>8}{'RMSE':>8}{'r':>7}"
          f"{'  within +/-3':>14}")
    print("-" * 68)
    all_bias = []
    for site in SITES:
        try:
            dates, tmmx = _read_series(site["name"], "tmmx")
            _, tmmn = _read_series(site["name"], "tmmn")
            _, rmax = _read_series(site["name"], "rmax")
            _, rmin = _read_series(site["name"], "rmin")
            _, vs = _read_series(site["name"], "vs")
            _, pr = _read_series(site["name"], "pr")
            _, srad = _read_series(site["name"], "srad")
            _, erc_ref = _read_series(site["name"], "erc")
            _, bi_ref = _read_series(site["name"], "bi")
        except FileNotFoundError as e:
            print(f"{site['name']:<14}  missing data -- run `fetch` first ({e})")
            continue

        m = min(len(tmmx), len(erc_ref))
        # --- unit conversions. Getting any of these wrong produces a smooth,
        # plausible, entirely wrong series.
        tmax_f = (tmmx[:m] - 273.15) * 9 / 5 + 32      # K -> F
        tmin_f = (tmmn[:m] - 273.15) * 9 / 5 + 32
        # 10 m -> 20 ft (6.1 m) via log profile over short vegetation,
        # then m/s -> mph.
        ws_mph = vs[:m] * 0.914 * 2.23694
        ppt_in = pr[:m] / 25.4                          # mm -> inches

        jdays = np.array([int(time.strftime(
            "%j", time.strptime(d[:10], "%Y-%m-%d"))) for d in dates[:m]])

        st = n.NFDRSState(1, climate_class=np.array([float(site["climat"])]))
        erc_ours = np.zeros(m)
        bi_ours = np.zeros(m)
        for i in range(m):
            sow = n.state_of_weather_from_srad(
                np.array([srad[i]]),
                np.array([_clear_sky_srad(site["lat"], jdays[i])]))
            out = n.step(st, dict(
                tmax_f=np.array([tmax_f[i]]), tmin_f=np.array([tmin_f[i]]),
                rhmax=np.array([rmax[:m][i]]), rhmin=np.array([rmin[:m][i]]),
                # 1300 LST sits near the daily extremes, so max temp and min
                # RH stand in for the observation-time values. This is the
                # standard gridded approximation and a prime suspect if the
                # bias comes back systematically dry.
                tobs_f=np.array([tmax_f[i]]), rhobs=np.array([rmin[:m][i]]),
                sow=sow, ppt_in=np.array([ppt_in[i]]),
                ws_mph=np.array([ws_mph[i]]),
                lat=np.array([site["lat"]]), jday=int(jdays[i])))
            erc_ours[i] = out["erc"][0]
            bi_ours[i] = out["bi"][0]

        s = slice(SPINUP_DAYS, m)
        a, b = erc_ours[s], erc_ref[:m][s]
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
        if len(a) < 100:
            print(f"{site['name']:<14}  too few overlapping days")
            continue
        bias = np.mean(a - b)
        mae = np.mean(np.abs(a - b))
        rmse = np.sqrt(np.mean((a - b) ** 2))
        r = np.corrcoef(a, b)[0, 1]
        within = 100.0 * np.mean(np.abs(a - b) <= 3)
        all_bias.append(bias)
        print(f"{site['name']:<14}{len(a):>6}{bias:>9.2f}{mae:>8.2f}"
              f"{rmse:>8.2f}{r:>7.3f}{within:>13.1f}%")

        np.savetxt(os.path.join(WORK, f"{site['name']}_erc_compare.csv"),
                   np.column_stack([a, b, a - b]), delimiter=",",
                   header="ours,gridmet,diff", comments="")

    if all_bias:
        print("-" * 68)
        print(f"mean bias across sites: {np.mean(all_bias):+.2f} ERC units\n")
        print("How to read this:")
        print("  r > 0.95 and |bias| < 3   engine is sound; proceed.")
        print("  r > 0.95 but bias large   constant offset. Suspect a unit")
        print("                            conversion or the 1300 LST proxy.")
        print("  r low, bias small         timing/state error. Suspect the")
        print("                            greenup rule or the 1000-h array.")
        print("  one site off, rest fine   that site's climate class is wrong.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    {"probe": probe, "fetch": fetch, "compare": compare}[cmd]()
