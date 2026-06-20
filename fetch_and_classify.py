#!/usr/bin/env python3
"""
Fetches current NIFC fire incidents and NASA FIRMS satellite hotspots for the
western US, classifies recurring (likely non-wildfire) heat sources based on
multi-day history, and writes the results to JSON files the Fireline website
can read directly as a static file — no personal API key or live browser
fetch required on the client side.
 
Intended to run on a schedule via GitHub Actions. Requires a FIRMS_MAP_KEY
environment variable (set as a GitHub Actions secret, never committed).
"""
import json
import time
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2
 
WESTERN_BBOX = (-125, 31, -102, 49.5)  # west, south, east, north
NIFC_POINTS_URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
HISTORY_PATH = "data/hotspot_history.json"
LATEST_PATH = "data/hotspots_latest.json"
LOOKBACK_DAYS = 14
REPEAT_THRESHOLD = 3  # distinct days seen before treated as a fixed source, not a fire
SUPPRESS_RADIUS_MILES = 3  # ignore detections this close to an already-tracked NIFC fire
 
 
def fetch_text(url, timeout=30, retries=3):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Fireline"}
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8")

        except Exception as e:
            if attempt == retries - 1:
                raise e

            print(f"Network error, retrying ({attempt+1}/{retries})...")
            time.sleep(10)
 
 
def fetch_json(url, timeout=30):
    return json.loads(fetch_text(url, timeout))
 
 
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
 
 
def fetch_known_incidents():
    xmin, ymin, xmax, ymax = WESTERN_BBOX
    params = {
        "where": "IncidentTypeCategory IN ('WF','CX')",
        "outFields": "IncidentName,InitialLatitude,InitialLongitude",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": "1500",
    }
    url = NIFC_POINTS_URL + "?" + urllib.parse.urlencode(params)
    gj = fetch_json(url)
    out = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        lat = coords[1] if coords else props.get("InitialLatitude")
        lon = coords[0] if coords else props.get("InitialLongitude")
        if lat and lon:
            out.append({"lat": float(lat), "lon": float(lon)})
    return out
 
 
def fetch_firms_points(map_key):
    xmin, ymin, xmax, ymax = WESTERN_BBOX
    url = f"{FIRMS_BASE}/{map_key}/VIIRS_SNPP_NRT/{xmin},{ymin},{xmax},{ymax}/1"
    text = fetch_text(url)
    if text.strip().startswith("<") or "invalid" in text.lower():
        raise RuntimeError("FIRMS returned an error page — check FIRMS_MAP_KEY is valid")
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    points = []
    for line in lines[1:]:
        cells = line.split(",")
        row = dict(zip(headers, cells))
        try:
            lat = float(row.get("latitude"))
            lon = float(row.get("longitude"))
            frp = float(row.get("frp") or 0)
        except (TypeError, ValueError):
            continue
        conf = (row.get("confidence") or "").strip().lower()
        date = (row.get("acq_date") or "").strip()
        if conf == "l":
            continue
        points.append({"lat": lat, "lon": lon, "frp": frp, "confidence": conf, "date": date})
    return points
 
 
def grid_key(lat, lon):
    return f"{round(lat*100)},{round(lon*100)}"
 
 
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {}
 
 
def prune_history(history):
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    pruned = {}
    for key, entry in history.items():
        dates = [
            d for d in entry.get("dates", [])
            if datetime.fromisoformat(d).replace(tzinfo=timezone.utc) >= cutoff
        ]
        if dates:
            pruned[key] = {"lat": entry["lat"], "lon": entry["lon"], "dates": dates}
    return pruned
 
 
def main():
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        print("FIRMS_MAP_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
 
    known = fetch_known_incidents()
    raw_points = fetch_firms_points(map_key)
 
    # Suppress detections within a few miles of a fire we already track from NIFC —
    # the unique value of this layer is catching heat that ISN'T officially logged yet.
    unmatched = [
        p for p in raw_points
        if not any(haversine(p["lat"], p["lon"], k["lat"], k["lon"]) < SUPPRESS_RADIUS_MILES for k in known)
    ]
 
    # De-duplicate overlapping satellite pixels onto a coarse grid, keeping the
    # highest-FRP point per cell so one fire doesn't produce a dozen entries.
    deduped = {}
    for p in unmatched:
        key = grid_key(p["lat"], p["lon"])
        if key not in deduped or p["frp"] > deduped[key]["frp"]:
            deduped[key] = p
 
    history = prune_history(load_history())
    classified = []
    for key, p in deduped.items():
        entry = history.setdefault(key, {"lat": p["lat"], "lon": p["lon"], "dates": []})
        if p["date"] and p["date"] not in entry["dates"]:
            entry["dates"].append(p["date"])
        days_seen = len(entry["dates"])
        classified.append({**p, "daysSeen": days_seen, "recurring": days_seen >= REPEAT_THRESHOLD})
 
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    with open(LATEST_PATH, "w") as f:
        json.dump({
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "hotspots": classified,
        }, f, indent=2)
 
    recurring_count = sum(1 for c in classified if c["recurring"])
    print(f"Wrote {len(classified)} hotspots ({recurring_count} recurring) to {LATEST_PATH}")
 
 
if __name__ == "__main__":
    main()
 