#!/usr/bin/env python3
"""
Fetches NOAA's experimental RRFS-SD (Rapid Refresh Forecast System — Smoke &
Dust) model output, decodes the near-surface smoke concentration field using
wgrib2, and writes small regridded PNG overlays (one per forecast hour) plus
a manifest.json describing them. The Fireline website reads these directly
as static files — no backend call needed on page load.

Intended to run on a schedule via GitHub Actions on a plain Ubuntu runner
(wgrib2 installed via apt — this is NOT feasible on most serverless
platforms, which is why this lives here instead of in a Vercel function).

RRFS-SD is still "parallel"/pre-operational data — NOAA has not published a
GRIB index (.idx) file for it, so byte-range partial downloads aren't
possible; we have to pull each full GRIB2 file (~25-40MB) and decode
locally. That's fine for a scheduled job, but would be too slow/expensive
for an on-demand serverless function, which is the other reason this is a
GitHub Actions job rather than a Vercel one.

IMPORTANT — things that are educated guesses, not confirmed:
  - Which file variant ("hi" vs "pr") contains the smoke variable
  - The exact GRIB field name for near-surface smoke mass density
This script tries several candidates and, on the first successful run,
prints (and can be read back from the Actions log) exactly what worked so
this comment block can be corrected. If ALL candidates fail for an hour, it
logs the full wgrib2 inventory for that file so a human can see what's
actually in there, then skips that hour rather than crashing the whole run.
"""
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/para"
OUT_DIR = "data/rrfs_smoke"
MANIFEST_PATH = f"{OUT_DIR}/manifest.json"

# Forecast hours to pull each run — every 6h out to 84h. Keeps bandwidth and
# runtime reasonable (15 files) while still covering the full RRFS-SD range.
FORECAST_HOURS = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]

# Candidate (file_variant, grib_field_match) pairs to try per forecast hour,
# in order, until one succeeds. wgrib2 -match uses simple substring matching
# against its inventory line, so these can be partial/likely names.
FIELD_CANDIDATES = [
    ("pr", ":MASSDEN:"),
    ("pr", ":COLMD:"),
    ("hi", ":MASSDEN:"),
    ("hi", ":COLMD:"),
]

# Western US bounds — matches OVERLAY_BOUNDS in index.html
LON_MIN, LON_MAX = -125.0, -102.0
LAT_MIN, LAT_MAX = 31.0, 49.0
GRID_DLON, GRID_DLAT = 0.05, 0.05  # ~5.5km pixels — modest size, fast to decode
GRID_NX = int(round((LON_MAX - LON_MIN) / GRID_DLON))
GRID_NY = int(round((LAT_MAX - LAT_MIN) / GRID_DLAT))

# Same µg/m³ bins/colors as the NWS Smoke Forecast layer, for visual consistency
COLOR_STOPS = [
    (0, (255, 255, 163)),
    (3, (250, 209, 87)),
    (25, (242, 166, 44)),
    (63, (171, 82, 19)),
    (158, (105, 0, 0)),
]


def http_ok(url, timeout=15):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Fireline"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def find_latest_cycle():
    """Find the most recent RRFS run that has actually finished publishing
    (checked by confirming its f084 file exists), stepping backward through
    cycles/days if the newest one isn't ready yet."""
    now = datetime.now(timezone.utc)
    for days_back in range(0, 3):
        date = now - timedelta(days=days_back)
        date_str = date.strftime("%Y%m%d")
        for hh in range(23, -1, -1):
            if days_back == 0 and hh > now.hour:
                continue
            cycle = f"{hh:02d}"
            check_url = (
                f"{NOMADS_BASE}/rrfs.{date_str}/{cycle}/"
                f"rrfs.t{cycle}z.2dfld.2p5km.f084.pr.grib2"
            )
            if http_ok(check_url):
                print(f"Found complete cycle: {date_str} {cycle}Z")
                return date_str, cycle
        print(f"No complete cycle found for {date_str}, trying earlier day...")
    raise RuntimeError("Could not find a complete RRFS-SD cycle in the last 3 days")


def download(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Fireline"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())


def wgrib2_inventory(grib_path):
    result = subprocess.run(
        ["wgrib2", "-s", grib_path], capture_output=True, text=True, timeout=60
    )
    return result.stdout


def extract_and_regrid(grib_path, field_match, out_grib_path):
    """Extract the matching record and regrid it to a simple regular lat/lon
    grid over our western-US bounds, in one wgrib2 call."""
    new_grid_spec = f"latlon {LON_MIN}:{GRID_NX}:{GRID_DLON} {LAT_MIN}:{GRID_NY}:{GRID_DLAT}"
    cmd = [
        "wgrib2", grib_path,
        "-match", field_match,
        "-new_grid_winds", "earth",
        "-new_grid_interpolation", "neighbor",
        "-new_grid", *new_grid_spec.split(),
        out_grib_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0 or not os.path.exists(out_grib_path):
        return False
    return os.path.getsize(out_grib_path) > 0


def grib_to_values(regridded_grib_path, text_out_path):
    """Dump the regridded single-field GRIB2 to a flat text grid wgrib2
    understands as: first line 'nx ny', then nx*ny values one per line."""
    result = subprocess.run(
        ["wgrib2", regridded_grib_path, "-text", text_out_path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not os.path.exists(text_out_path):
        return None
    with open(text_out_path) as f:
        lines = f.read().split()
    if len(lines) < 2:
        return None
    nx, ny = int(float(lines[0])), int(float(lines[1]))
    values = [float(v) for v in lines[2:2 + nx * ny]]
    if len(values) != nx * ny:
        print(f"  WARNING: expected {nx*ny} values, got {len(values)}")
        return None
    return nx, ny, values


def value_to_color(v):
    if v is None or v != v or v < 0:  # NaN check via v != v
        return (0, 0, 0, 0)  # transparent
    for i in range(len(COLOR_STOPS) - 1, -1, -1):
        threshold, color = COLOR_STOPS[i]
        if v >= threshold:
            return (*color, 140)  # ~55% opacity, matches other Fireline overlays
    return (0, 0, 0, 0)


def render_png(nx, ny, values, out_png_path):
    from PIL import Image
    img = Image.new("RGBA", (nx, ny))
    pixels = img.load()
    for row in range(ny):
        for col in range(nx):
            v = values[row * nx + col]
            # wgrib2's regular lat/lon grid (lat0:ny:dlat with lat0 = southern
            # edge) writes south-to-north; flip vertically so row 0 = north,
            # matching standard image coordinates (top = north).
            pixels[col, ny - 1 - row] = value_to_color(v)
    img.save(out_png_path)


def process_hour(date_str, cycle, fhour, workdir):
    fhour_str = f"{fhour:03d}"
    print(f"\n--- Forecast hour f{fhour_str} ---")
    for variant, field_match in FIELD_CANDIDATES:
        url = (
            f"{NOMADS_BASE}/rrfs.{date_str}/{cycle}/"
            f"rrfs.t{cycle}z.2dfld.2p5km.f{fhour_str}.{variant}.grib2"
        )
        grib_path = f"{workdir}/f{fhour_str}.{variant}.grib2"
        try:
            if not os.path.exists(grib_path):
                print(f"  Downloading {variant} file...")
                download(url, grib_path)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  Could not download {variant} file: {e}")
            continue

        inv = wgrib2_inventory(grib_path)
        if field_match.strip(":") not in inv:
            print(f"  Field {field_match} not found in {variant} file, trying next candidate")
            continue

        regridded = f"{workdir}/f{fhour_str}.regridded.grib2"
        if not extract_and_regrid(grib_path, field_match, regridded):
            print(f"  Extract/regrid failed for {field_match} in {variant} file")
            continue

        text_out = f"{workdir}/f{fhour_str}.txt"
        parsed = grib_to_values(regridded, text_out)
        if not parsed:
            print(f"  Could not parse regridded output for f{fhour_str}")
            continue

        nx, ny, values = parsed
        png_path = f"{OUT_DIR}/f{fhour_str}.png"
        render_png(nx, ny, values, png_path)
        print(f"  SUCCESS using variant={variant} field={field_match} -> {png_path}")
        return {"hour": fhour, "file": f"f{fhour_str}.png", "variant": variant, "field": field_match}

    # Every candidate failed for this hour — log full inventory of the last
    # downloaded file so a human can see what fields actually exist.
    print(f"  ALL candidates failed for f{fhour_str}. Full inventory of last file tried:")
    print(inv if 'inv' in dir() else "  (no file could be downloaded)")
    return None


def main():
    date_str, cycle = find_latest_cycle()
    os.makedirs(OUT_DIR, exist_ok=True)
    workdir = "/tmp/rrfs_smoke_work"
    os.makedirs(workdir, exist_ok=True)

    results = []
    for fhour in FORECAST_HOURS:
        try:
            r = process_hour(date_str, cycle, fhour, workdir)
            if r:
                results.append(r)
        except Exception as e:
            print(f"Unexpected error on f{fhour:03d}: {e}", file=sys.stderr)
            continue

    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "cycleDate": date_str,
        "cycleHour": cycle,
        "bounds": [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        "hours": results,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(results)}/{len(FORECAST_HOURS)} forecast hours to {MANIFEST_PATH}")
    if not results:
        print("WARNING: zero hours succeeded — check the inventory logs above", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
