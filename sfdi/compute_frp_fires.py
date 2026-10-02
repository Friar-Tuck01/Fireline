"""
compute_frp_fires.py -- per-fire observed Fire Radiative Power -> data/frp_latest.json

Why this runs server-side instead of in the browser:

index.html already loads FIRMS detections, but that layer deliberately DISCARDS
everything within ~3 miles of a tracked fire -- its job is catching heat nobody
has logged yet -- and then grid-snaps the survivors keeping only the highest-FRP
point per cell. Both steps are right for that layer and fatal for this one: the
detections a fire produced are exactly the ones removed, and the magnitudes of
what remains have already been collapsed. Per-fire FRP has to come from the raw
detections, so it is computed here, once, the same way PFT is.

The assignment rules, and why they are not a radius query, are documented in
frp_assign.py. Short version: every detection goes to at most one fire, chosen
by how deep inside that fire's own footprint it sits, so two nearby fires can
never both be credited with the same heat.

    python compute_frp_fires.py --out data/frp_latest.json

Needs a FIRMS MAP_KEY (free, via a NASA Earthdata login, at
https://firms.modaps.eosdis.nasa.gov/api/map_key/). Pass --key or set
FIRMS_MAP_KEY. In CI it belongs in repository secrets, never in the page --
a key in index.html is a key published to everyone who views it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from frp_assign import (PERIM_BUFFER_KM, assign_detections, claim_radius_km,
                        geom_area_acres, haversine_km, perimeter_is_plausible,
                        summarize_fire)

POINTS_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/"
              "services/WFIGS_Incident_Locations_Current/FeatureServer/0/query")
# Same perimeter service index.html draws. A real polygon beats any circle
# inferred from a point and an acreage, so detections are tested against these
# first and only fall back to the circle where no perimeter is published.
PERIM_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/"
             "services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query")
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
WESTERN_BBOX = {"xmin": -125, "ymin": 31, "xmax": -102, "ymax": 49.5}

# Both operational VIIRS platforms. More platforms means more overpasses, which
# means a better chance of a recent one -- they are never summed together, only
# compared, so adding a satellite cannot inflate a fire's reported power.
SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT")

USER_AGENT = "fireline-frp/1.0 (+https://github.com/Friar-Tuck01/Fireline)"


def fetch_fires(timeout=60):
    """
    Current WFIGS incidents. The id is built exactly as index.html builds it,
    so the popup can look a fire up directly; if the two ever diverge the panel
    silently shows nothing.

    Prescribed burns are included as CLAIMANTS on purpose. An RX burn puts real
    heat on the satellite, and if it cannot claim its own detections they get
    handed to whatever wildfire is next-nearest.
    """
    params = {
        "where": "IncidentTypeCategory IN ('WF','CX','RX')",
        "outFields": ("IncidentName,IncidentSize,IncidentTypeCategory,"
                      "InitialLatitude,InitialLongitude,UniqueFireIdentifier"),
        "geometry": (f"{WESTERN_BBOX['xmin']},{WESTERN_BBOX['ymin']},"
                     f"{WESTERN_BBOX['xmax']},{WESTERN_BBOX['ymax']}"),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326", "f": "geojson", "resultRecordCount": "1500",
    }
    url = POINTS_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        gj = json.loads(r.read().decode("utf-8"))

    fires = []
    for f in gj.get("features", []):
        p = f.get("properties") or {}
        coords = (f.get("geometry") or {}).get("coordinates") or []
        lon = coords[0] if len(coords) > 1 else p.get("InitialLongitude")
        lat = coords[1] if len(coords) > 1 else p.get("InitialLatitude")
        if lat is None or lon is None:
            continue
        fid = p.get("UniqueFireIdentifier") or f"{p.get('IncidentName')}{lat}{lon}"
        fires.append({
            "id": fid,
            "name": p.get("IncidentName") or "Unnamed Incident",
            "lat": float(lat), "lon": float(lon),
            "acres": p.get("IncidentSize") or 0,
            "category": p.get("IncidentTypeCategory"),
        })
    return fires


def fetch_perimeters(timeout=120):
    """
    Current WFIGS perimeters, keyed by the same UniqueFireIdentifier the
    points use so the two line up.

    maxAllowableOffset simplifies the geometry server-side. Full-resolution
    perimeters for a whole fire season run to tens of megabytes, and a
    vertex-level fidelity of ~100 m is far finer than a 375 m VIIRS pixel --
    paying for more detail than the detections can resolve would be waste.
    """
    params = {
        "where": "1=1",
        "outFields": "attr_UniqueFireIdentifier",
        "geometry": (f"{WESTERN_BBOX['xmin']},{WESTERN_BBOX['ymin']},"
                     f"{WESTERN_BBOX['xmax']},{WESTERN_BBOX['ymax']}"),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326", "f": "geojson", "resultRecordCount": "2000",
        "maxAllowableOffset": "0.001",     # ~100 m at these latitudes
    }
    url = PERIM_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        gj = json.loads(r.read().decode("utf-8"))

    out = {}
    for f in gj.get("features", []):
        fid = ((f.get("properties") or {}).get("attr_UniqueFireIdentifier"))
        geom = f.get("geometry")
        if not fid or not geom:
            continue
        # A fire can publish several perimeter records. Keep the one with the
        # most vertices, which is the most complete mapping of it.
        prev = out.get(fid)
        if prev is None or _vertex_count(geom) > _vertex_count(prev):
            out[fid] = geom
    return out


def _vertex_count(geom):
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        return sum(len(r) for r in c)
    if t == "MultiPolygon":
        return sum(len(r) for poly in c for r in poly)
    return 0


def fetch_firms(map_key, source, day_range=1, timeout=120):
    """One FIRMS source over the western bbox. Returns parsed detections."""
    bbox = (f"{WESTERN_BBOX['xmin']},{WESTERN_BBOX['ymin']},"
            f"{WESTERN_BBOX['xmax']},{WESTERN_BBOX['ymax']}")
    url = f"{FIRMS_BASE}/{map_key}/{source}/{bbox}/{day_range}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", errors="replace")

    # FIRMS answers a bad key with an HTML page and HTTP 200, so a naive
    # csv parse would quietly yield zero rows and look like "no fires burning".
    head = text.lstrip()[:200].lower()
    if head.startswith("<") or "invalid" in head:
        raise RuntimeError(f"{source}: FIRMS returned an error page "
                           f"(usually an invalid or expired MAP_KEY)")

    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        conf = (row.get("confidence") or "").strip().lower()
        if conf == "l":       # same low-confidence filter index.html applies
            continue
        out.append({
            "lat": lat, "lon": lon,
            "frp": float(row.get("frp") or 0.0),
            "acq_date": (row.get("acq_date") or "").strip(),
            "acq_time": (row.get("acq_time") or "").strip(),
            # Tag by the source we asked for, not the CSV's own satellite
            # column -- that column's spelling has changed between versions.
            "satellite": source,
        })
    return out


def _age_hours(det, _now=None):
    """Hours since a detection's acquisition, or None if unparseable."""
    now = _now or dt.datetime.now(dt.timezone.utc)
    raw = "".join(ch for ch in str(det.get("acq_time") or "") if ch.isdigit())
    raw = raw.rjust(4, "0")
    try:
        y, mo, dy = (int(x) for x in str(det.get("acq_date") or "").split("-"))
        t = dt.datetime(y, mo, dy, int(raw[:-2]), int(raw[-2:]),
                        tzinfo=dt.timezone.utc)
    except (ValueError, TypeError):
        return None
    return (now - t).total_seconds() / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/frp_latest.json")
    ap.add_argument("--key", default=os.environ.get("FIRMS_MAP_KEY", ""))
    # Two days, not one. FIRMS day_range counts whole UTC DAYS, not rolling
    # hours: day_range=1 means "since 00:00 UTC today", so a run just after
    # UTC midnight sees an almost empty window. That is not hypothetical --
    # it failed a run at 00:08 UTC with zero detections from both platforms,
    # and the 00:40 UTC cron would have hit it every single night.
    #
    # Fetching two days and then trimming to a true rolling window below
    # makes the result independent of where the run lands in the UTC day.
    ap.add_argument("--days", type=int, default=2,
                    help="FIRMS lookback in whole UTC days (1-10). Fetched "
                         "wide, then trimmed to --hours.")
    ap.add_argument("--hours", type=float, default=24.0,
                    help="Rolling window actually used, in hours back from now.")
    args = ap.parse_args()

    if not args.key:
        sys.exit("No FIRMS MAP_KEY. Pass --key or set FIRMS_MAP_KEY. "
                 "Free via NASA Earthdata: "
                 "https://firms.modaps.eosdis.nasa.gov/api/map_key/")

    print("Fireline FRP -- per-fire observed Fire Radiative Power")
    print("  Exclusive assignment: no fire can be credited with another's heat\n")

    print("Fetching active fires from WFIGS ...")
    fires = fetch_fires()
    print(f"  {len(fires)} fires in the western domain")

    # A missing perimeter feed must not void the run -- the claim-radius
    # fallback still produces a usable answer, and the output records how many
    # detections rested on a real polygon so the drop is visible.
    try:
        perims = fetch_perimeters()
        known = {f["id"] for f in fires}
        matched = sum(1 for k in perims if k in known)
        # Report perimeters whose size disagrees with the reported acreage.
        # They are ignored in favour of the circle, and saying so is the only
        # way this stays visible -- the alternative is a confident wrong
        # number carrying an "inside perimeter" marker.
        acres = {f["id"]: f.get("acres") for f in fires}
        rejected = [(k, geom_area_acres(perims[k]), acres.get(k))
                    for k in perims if k in known
                    and not perimeter_is_plausible(perims[k], acres.get(k))]
        print(f"  {len(perims)} perimeters, {matched} matching a current fire")
        if rejected:
            print(f"  {len(rejected)} perimeter(s) ignored as implausibly "
                  f"large for the reported acreage:")
            for k, a, rep in rejected[:5]:
                print(f"      {k}: polygon ~{a:,.0f} acres vs reported "
                      f"{rep or 0:,.0f}")
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as e:
        perims = {}
        print(f"  perimeters FAILED ({e}) -- falling back to claim radii")

    dets = []
    for src in SOURCES:
        try:
            got = fetch_firms(args.key, src, args.days)
            dets.extend(got)
            print(f"  {src}: {len(got)} detections")
        except (urllib.error.URLError, RuntimeError) as e:
            # One platform down must not void the run; the other still gives a
            # usable answer, and the output records which were used.
            print(f"  {src}: FAILED -- {e}")
    if not dets:
        sys.exit("No detections from any source over the whole fetch window; "
                 "refusing to write an empty file. Check the FIRMS key and "
                 "whether the service is up -- over a multi-day window this "
                 "should not happen even in a quiet fire season.")

    # Trim the whole-UTC-day fetch down to a true rolling window, so the
    # numbers mean the same thing whatever time of day the job runs.
    before = len(dets)
    dets = [d for d in dets if _age_hours(d) is not None
            and _age_hours(d) <= args.hours]
    print(f"  {len(dets)} within the last {args.hours:g} h "
          f"(trimmed {before - len(dets)} older)")
    if not dets:
        sys.exit(f"Nothing in the last {args.hours:g} h; refusing to write.")

    print(f"\nAssigning {len(dets)} detections to {len(fires)} fires ...")
    assigned, unassigned = assign_detections(fires, dets, perims)

    by_id = {f["id"]: f for f in fires}
    out_fires, n_contested_tot = {}, 0
    for fid, dlist in assigned.items():
        s = summarize_fire(dlist)
        if not s:
            continue
        f = by_id[fid]
        # Distance to the nearest OTHER fire, so the panel can say when a
        # number sits close enough to a neighbour to deserve caution.
        others = [(haversine_km(f["lat"], f["lon"], g["lat"], g["lon"]), g["name"])
                  for g in fires if g["id"] != fid]
        near_km, near_name = min(others) if others else (None, None)
        rec = {"name": f["name"], **s}
        # Only meaningful when the circle was actually used; stating a claim
        # radius for a fire assigned from its own polygon would imply the
        # circle had something to do with the answer.
        if not s["n_in_perim"] and not s.get("n_near_perim"):
            rec["claim_km"] = round(claim_radius_km(f["acres"]), 1)
        if near_km is not None and near_km < 50:
            rec["nearest_km"] = round(near_km, 1)
            rec["nearest_name"] = near_name
        n_contested_tot += s["n_contested"]
        out_fires[fid] = rec

    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "NASA FIRMS VIIRS NRT (" + ", ".join(SOURCES) + ")",
        "units": "MW",
        "window_hours": args.hours,
        "fetch_days": args.days,
        "assignment": ("exclusive nearest-footprint; each detection credited to "
                       "at most one fire; see frp_assign.py"),
        "caveat": ("Radiative power only -- roughly 10-15% of a fire's total "
                   "heat release, and not directly comparable with PFT, which "
                   "is a threshold on total convective heat flux in GW."),
        "n_fires": len(fires),
        "n_with_frp": len(out_fires),
        "n_detections": len(dets),
        "n_assigned": sum(len(v) for v in assigned.values()),
        "n_unassigned": len(unassigned),
        "n_contested": n_contested_tot,
        "n_perimeters": len(perims),
        "n_from_perimeter": sum(1 for r in out_fires.values()
                                if r.get("n_in_perim")),
        "n_with_flank_detections": sum(1 for r in out_fires.values()
                                       if r.get("n_near_perim")),
        "perimeter_buffer_km": PERIM_BUFFER_KM,
        "fires": out_fires,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    size = os.path.getsize(args.out)
    print(f"\nWrote {args.out}  ({size / 1024:.1f} KB)")
    print(f"  {len(out_fires)} fires with a value, "
          f"{len(fires) - len(out_fires)} without")
    print(f"  {payload['n_assigned']} detections assigned, "
          f"{len(unassigned)} unassigned (heat with no logged fire)")
    print(f"  {n_contested_tot} detections contested by a neighbouring fire")
    n_perim = sum(1 for r in out_fires.values() if r.get("n_in_perim"))
    n_near = sum(1 for r in out_fires.values() if r.get("n_near_perim"))
    print(f"  {n_perim} fires assigned from a published perimeter, "
          f"{len(out_fires) - n_perim} from a claim radius")
    print(f"  {n_near} fires had detections just beyond the mapped line "
          f"(within {PERIM_BUFFER_KM:g} km) -- spreading flank, stale perimeter")
    vals = sorted(r["frp_mw"] for r in out_fires.values())
    if vals:
        def pct(p):
            return vals[min(len(vals) - 1, int(len(vals) * p / 100))]
        print(f"  FRP MW  p10={pct(10):.0f}  median={pct(50):.0f}  "
              f"p90={pct(90):.0f}  max={vals[-1]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
