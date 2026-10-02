"""
frp_assign.py -- assign satellite fire detections to fires, exclusively.

The whole difficulty is the one the brief named: when two fires burn near each
other, neither may be credited with the other's heat. A radius query ("sum
everything within N km") cannot satisfy that -- any detection in the overlap
is counted twice, and the error grows exactly where it matters most, in
multi-fire complexes.

So assignment here is EXCLUSIVE: every detection is given to at most one fire.

Two things make that non-trivial, and both are handled below.

1. Nearest centroid is the wrong rule when fires differ in size.
   A detection 4 km from a 50-acre fire and 10 km from a 370,000-acre fire is
   nearer the small one, but it lies far outside the small fire's footprint
   (radius ~0.2 km) and deep inside the big one's (~22 km). Crediting the
   small fire would multiply its reported power many times over.
   Instead each fire gets a claim radius from its own size, and detections are
   scored by NORMALISED distance, d / claim_radius -- "how far into this
   fire's own footprint does it sit". Lowest score wins. A fire can only claim
   what is plausibly inside it.

2. Summing a whole day double-counts the same fire.
   FRP is a rate (megawatts), not a quantity. Suomi-NPP and NOAA-20 each pass
   roughly twice a day, so adding every detection in 24 h sums four snapshots
   of one fire and reports ~4x its actual power. Detections are therefore
   grouped into overpasses and each overpass summed separately.

Both functions are pure, so they can be tested without network access.
"""

from __future__ import annotations

import math
from collections import defaultdict

# A VIIRS pixel is ~375 m and geolocation error is of similar order; a
# perimeter is also hours stale by the time a satellite sees the fire. This
# buffer covers both, so a detection just outside the mapped footprint is not
# orphaned.
CLAIM_BUFFER_KM = 2.0
# Floor: WFIGS reports many fires at 0 or 0.1 acres for hours after discovery,
# and a new fire is exactly when its heat matters. Without a floor those fires
# could claim nothing at all.
CLAIM_MIN_KM = 3.0
# Ceiling: stops one megafire's claim from stretching across a whole GACC and
# swallowing unrelated fires. 25 km is already larger than most fires.
CLAIM_MAX_KM = 25.0
# If the runner-up's score is within this factor of the winner's, the
# detection was nearly claimed by a neighbour. It is still assigned to exactly
# one fire -- never split, never shared -- but it is counted as contested so
# the display can say the number is uncertain rather than pretending it isn't.
CONTEST_RATIO = 1.25
# How far beyond a mapped perimeter a detection can sit and still belong to
# that fire. Covers the VIIRS pixel itself (~375 m), geolocation error of
# similar order, and growth since the perimeter was last mapped -- a
# perimeter can be six or eight hours old when the satellite passes, and an
# active flank moves in that time. Kept modest because the buffer is applied
# outward in every direction: large enough to catch a spreading edge, small
# enough that it cannot reach across to a neighbouring fire's heat.
PERIM_BUFFER_KM = 2.0
# A VIIRS swath crosses a single fire in well under a minute; successive
# overpasses are hours apart. Anything over this gap is a different pass.
OVERPASS_GAP_MIN = 20.0

ACRE_M2 = 4046.8564224


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def claim_radius_km(acres):
    """
    Radius of the circle with the same area as the fire, plus slack.

    A real fire is not a circle, but WFIGS gives a point and an acreage and
    nothing about shape, so an equal-area circle is the most the input
    supports. Claiming more would be inventing geometry.
    """
    try:
        a = max(float(acres or 0.0), 0.0)
    except (TypeError, ValueError):
        a = 0.0
    r_km = math.sqrt(a * ACRE_M2 / math.pi) / 1000.0
    return min(CLAIM_MAX_KM, max(CLAIM_MIN_KM, r_km + CLAIM_BUFFER_KM))


def _point_in_ring(lon, lat, ring):
    """Ray casting over a GeoJSON ring, which is ordered [lon, lat]."""
    inside = False
    n = len(ring)
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[i - 1][0], ring[i - 1][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


def point_in_geometry(lon, lat, geom):
    """
    Point in a GeoJSON Polygon or MultiPolygon, holes respected.

    Perimeters genuinely have holes -- unburned islands inside the fire
    boundary -- and a detection in one is not inside the fire.
    """
    if not geom:
        return False
    kind = geom.get("type")
    if kind == "Polygon":
        polys = [geom.get("coordinates") or []]
    elif kind == "MultiPolygon":
        polys = geom.get("coordinates") or []
    else:
        return False
    for rings in polys:
        if not rings or not _point_in_ring(lon, lat, rings[0]):
            continue
        if any(_point_in_ring(lon, lat, h) for h in rings[1:]):
            continue        # inside a hole, so outside the fire
        return True
    return False


def _geom_bbox(geom):
    """(min_lon, min_lat, max_lon, max_lat), or None."""
    pts = []
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        for r in c:
            pts.extend(r)
    elif t == "MultiPolygon":
        for poly in c:
            for r in poly:
                pts.extend(r)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _seg_dist_km(px, py, ax, ay, bx, by, kx, ky):
    """
    Distance from a point to a segment, in km, on a local flat projection.

    kx/ky convert degrees to km at this latitude. Over the few km that matter
    here the error from ignoring curvature is centimetres, and a projection
    keeps this cheap enough to run over every perimeter edge.
    """
    px, ax, bx = px * kx, ax * kx, bx * kx
    py, ay, by = py * ky, ay * ky, by * ky
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_geometry_km(lon, lat, geom):
    """
    Shortest distance from a point to a polygon's boundary, 0.0 if inside.

    This exists because a perimeter is a snapshot. It is flown or mapped at
    one moment and can be six or eight hours old by the time a satellite
    passes, during which an active fire keeps moving. Heat just beyond the
    mapped line is the NORMAL appearance of a spreading flank, not a
    different fire, and dropping it would bias the figure down hardest on
    exactly the fires that are running.
    """
    if point_in_geometry(lon, lat, geom):
        return 0.0
    kx = 111.320 * math.cos(math.radians(lat))
    ky = 110.574
    t, c = geom.get("type"), geom.get("coordinates") or []
    rings = c if t == "Polygon" else [r for poly in c for r in poly] \
        if t == "MultiPolygon" else []
    best = float("inf")
    for ring in rings:
        for i in range(1, len(ring)):
            ax, ay = ring[i - 1][0], ring[i - 1][1]
            bx, by = ring[i][0], ring[i][1]
            d = _seg_dist_km(lon, lat, ax, ay, bx, by, kx, ky)
            if d < best:
                best = d
    return best


def ring_area(geom):
    """
    Rough area of a geometry in square degrees, longitude scaled by latitude.

    Used only to break ties when a detection falls inside more than one
    perimeter -- complexes publish a perimeter that contains their member
    fires' perimeters, so nesting is normal. The smallest containing polygon
    is the most specific answer, and only the ordering matters here, not the
    absolute number.
    """
    if not geom:
        return float("inf")
    kind = geom.get("type")
    polys = ([geom.get("coordinates") or []] if kind == "Polygon"
             else geom.get("coordinates") or [])
    total = 0.0
    for rings in polys:
        if not rings:
            continue
        outer = rings[0]
        if len(outer) < 3:
            continue
        lat0 = sum(p[1] for p in outer) / len(outer)
        scale = math.cos(math.radians(lat0)) or 1.0
        s = 0.0
        for i in range(len(outer)):
            x1, y1 = outer[i - 1][0] * scale, outer[i - 1][1]
            x2, y2 = outer[i][0] * scale, outer[i][1]
            s += x1 * y2 - x2 * y1
        total += abs(s) / 2.0
    return total or float("inf")


def assign_detections(fires, detections, perimeters=None):
    """
    Give each detection to at most one fire.

    fires:      [{"id","lat","lon","acres", ...}]
    detections: [{"lat","lon","frp", ...}]
    perimeters: optional {fire id -> GeoJSON geometry}

    Returns (assignments, unassigned) where assignments maps fire id ->
    list of detections, each carrying "_score", "_dist_km", "_contested" and
    "_in_perim".

    Two stages, in order of how much the data actually supports:

      1. If a mapped perimeter CONTAINS the detection, that fire gets it. This
         is the strongest evidence available -- a real polygon, not a guess at
         shape -- and it settles the adjacent-fire problem outright, because
         perimeters do not overlap except where a complex nests its own
         members, which the smallest-polygon rule resolves.

      2. Otherwise fall back to the claim-radius rule below. Most fires have
         no published perimeter (small, new, or simply not yet flown), and a
         perimeter is in any case hours old by the time a satellite passes, so
         heat just outside it is expected and should not be orphaned.

    Guarantee, in both stages: sum of assigned + unassigned == len(detections),
    and no detection appears under two fires. The tests assert this.
    """
    perimeters = perimeters or {}
    known_ids = {f["id"] for f in fires}
    # Pre-size and pre-box the perimeters once rather than per detection. The
    # bbox test is what keeps the distance pass affordable: without it this
    # would measure every detection against every edge of every polygon.
    perim_list = [(fid, g, ring_area(g), _geom_bbox(g))
                  for fid, g in perimeters.items()
                  if g and fid in known_ids]
    perim_list = [p for p in perim_list if p[3]]

    prepared = [(f, claim_radius_km(f.get("acres"))) for f in fires]
    assignments = defaultdict(list)
    unassigned = []

    for d in detections:
        # --- stage 1: inside a published perimeter ------------------------
        hits = [(area, fid) for fid, g, area, _bb in perim_list
                if point_in_geometry(d["lon"], d["lat"], g)]
        if hits:
            hits.sort()                      # smallest containing polygon wins
            rec = dict(d)
            rec["_in_perim"] = True
            rec["_near_perim"] = False
            rec["_score"] = 0.0
            rec["_dist_km"] = 0.0
            # A perimeter containing the point is not a near miss, whatever
            # else is nearby, so this is never contested.
            rec["_contested"] = False
            assignments[hits[0][1]].append(rec)
            continue

        # --- stage 1b: just outside a perimeter ---------------------------
        # The spreading-flank case. Nearest boundary wins, so a detection
        # between two fires still goes to exactly one of them.
        near = None
        lat_deg = PERIM_BUFFER_KM / 110.574
        lon_deg = PERIM_BUFFER_KM / max(
            111.320 * math.cos(math.radians(d["lat"])), 1e-6)
        for fid, g, _area, bb in perim_list:
            if (d["lon"] < bb[0] - lon_deg or d["lon"] > bb[2] + lon_deg or
                    d["lat"] < bb[1] - lat_deg or d["lat"] > bb[3] + lat_deg):
                continue
            dist = distance_to_geometry_km(d["lon"], d["lat"], g)
            if dist <= PERIM_BUFFER_KM and (near is None or dist < near[0]):
                near = (dist, fid)
        if near:
            rec = dict(d)
            rec["_in_perim"] = False
            rec["_near_perim"] = True
            rec["_score"] = 0.0
            rec["_dist_km"] = round(near[0], 2)
            rec["_contested"] = False
            assignments[near[1]].append(rec)
            continue

        # --- stage 2: claim radius ---------------------------------------
        best = None, float("inf"), None   # fire, score, dist
        second = float("inf")
        for f, radius in prepared:
            dist = haversine_km(d["lat"], d["lon"], f["lat"], f["lon"])
            score = dist / radius
            if score < best[1]:
                second = best[1]
                best = f, score, dist
            elif score < second:
                second = score

        fire, score, dist = best
        if fire is None or score > 1.0:
            unassigned.append(d)
            continue

        rec = dict(d)
        rec["_in_perim"] = False
        rec["_near_perim"] = False
        rec["_score"] = score
        rec["_dist_km"] = dist
        # Contested only if the runner-up could ALSO have legitimately claimed
        # it (score <= 1). A distant fire being 1.1x worse than the winner is
        # not a real contest if the detection is outside its footprint anyway.
        rec["_contested"] = second <= 1.0 and second <= score * CONTEST_RATIO
        assignments[fire["id"]].append(rec)

    return dict(assignments), unassigned


def _minutes(det):
    """Detection timestamp in minutes, from acq_date + acq_time."""
    date = str(det.get("acq_date") or "").strip()
    raw = str(det.get("acq_time") or "0").strip()
    digits = "".join(ch for ch in raw if ch.isdigit()).rjust(4, "0")
    hh, mm = int(digits[:-2] or 0), int(digits[-2:] or 0)
    try:
        y, mo, dy = (int(x) for x in date.split("-"))
    except ValueError:
        y, mo, dy = 1970, 1, 1
    days = (y * 372) + (mo * 31) + dy      # monotonic enough to order by
    return days * 1440 + hh * 60 + mm


def group_overpasses(detections):
    """
    Split one fire's detections into satellite overpasses.

    Summing across passes would report a fire's power as the total of several
    separate snapshots hours apart. Each satellite is grouped on its own,
    because SNPP and NOAA-20 see the same fire minutes to hours apart and
    those are genuinely different observations of the same thing, not one
    larger fire.

    Returns a list of (sort_key_minutes, satellite, [detections]) oldest first.
    """
    by_sat = defaultdict(list)
    for d in detections:
        by_sat[str(d.get("satellite") or "?")].append(d)

    passes = []
    for sat, dets in by_sat.items():
        dets = sorted(dets, key=_minutes)
        cur = [dets[0]]
        for prev, nxt in zip(dets, dets[1:]):
            if _minutes(nxt) - _minutes(prev) > OVERPASS_GAP_MIN:
                passes.append((_minutes(cur[0]), sat, cur))
                cur = [nxt]
            else:
                cur.append(nxt)
        passes.append((_minutes(cur[0]), sat, cur))

    passes.sort(key=lambda t: t[0])
    return passes


def summarize_fire(dets):
    """
    Reduce one fire's assigned detections to what the panel shows.

    frp_mw is the MOST RECENT overpass, not a 24 h total: FRP is a rate, and
    the latest pass is the only one that describes the fire now.
    """
    if not dets:
        return None
    passes = group_overpasses(dets)
    sums = [(sum(float(d.get("frp") or 0.0) for d in grp), key, sat, grp)
            for key, sat, grp in passes]
    latest_sum, _, latest_sat, latest_grp = sums[-1]
    peak = max(s for s, _, _, _ in sums)
    contested = sum(1 for d in latest_grp if d.get("_contested"))
    in_perim = sum(1 for d in latest_grp if d.get("_in_perim"))
    near_perim = sum(1 for d in latest_grp if d.get("_near_perim"))
    hottest = max(float(d.get("frp") or 0.0) for d in latest_grp)
    return {
        # Total radiative power of the fire on the most recent pass: the sum
        # of its pixels. This is the headline figure.
        "frp_mw": round(latest_sum, 1),
        # The single hottest pixel in that pass. A fire can total 400 MW as
        # forty quiet pixels or as one violent head, and those are different
        # situations; the total alone cannot tell them apart.
        "frp_max_pixel_mw": round(hottest, 1),
        # The largest single-pass total in the whole lookback window -- how
        # hot this fire has been recently, not how hot it is now.
        "frp_peak_mw": round(peak, 1),
        "n_det": len(latest_grp),
        "n_in_perim": in_perim,
        # Detections just beyond the mapped line -- a spreading flank the
        # perimeter has not caught up with, counted but distinguished.
        "n_near_perim": near_perim,
        "n_contested": contested,
        "n_overpasses": len(sums),
        "satellite": latest_sat,
        "acq_date": latest_grp[0].get("acq_date"),
        "acq_time": latest_grp[0].get("acq_time"),
    }
