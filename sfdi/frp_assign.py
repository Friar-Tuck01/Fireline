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


def assign_detections(fires, detections):
    """
    Give each detection to at most one fire.

    fires:      [{"id","lat","lon","acres", ...}]
    detections: [{"lat","lon","frp", ...}]

    Returns (assignments, unassigned) where assignments maps fire id ->
    list of detections, each carrying "_score", "_dist_km" and "_contested".

    Guarantee: sum of all assigned detections + unassigned == len(detections),
    and no detection appears under two fires. The tests assert this.
    """
    prepared = [(f, claim_radius_km(f.get("acres"))) for f in fires]
    assignments = defaultdict(list)
    unassigned = []

    for d in detections:
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
    return {
        "frp_mw": round(latest_sum, 1),
        "frp_peak_mw": round(peak, 1),
        "n_det": len(latest_grp),
        "n_contested": contested,
        "n_overpasses": len(sums),
        "satellite": latest_sat,
        "acq_date": latest_grp[0].get("acq_date"),
        "acq_time": latest_grp[0].get("acq_time"),
    }
