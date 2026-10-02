"""
Tests for exclusive detection assignment.

The brief was explicit: one fire must never absorb a neighbour's heat. These
tests are built around the ways that actually happens -- adjacent fires, a
megafire beside a new start, co-located complex members -- plus the invariant
that makes double-counting impossible by construction.
"""

import math
import unittest

from frp_assign import (CLAIM_MAX_KM, CLAIM_MIN_KM, assign_detections,
                        claim_radius_km, group_overpasses, haversine_km,
                        summarize_fire)

KM_PER_DEG_LAT = 110.574


def at(lat, lon, frp=10.0, date="2026-10-01", time="2112", sat="N"):
    return {"lat": lat, "lon": lon, "frp": frp,
            "acq_date": date, "acq_time": time, "satellite": sat}


def north_of(lat, lon, km):
    return lat + km / KM_PER_DEG_LAT, lon


class TestClaimRadius(unittest.TestCase):
    def test_equal_area_circle(self):
        # 5,667 acres (the Dome fire) -> ~2.7 km equal-area radius, + 2 km slack
        self.assertAlmostEqual(claim_radius_km(5667), 2.7 + 2.0, delta=0.1)

    def test_floor_applies_to_brand_new_fires(self):
        # WFIGS reports 0 acres for hours after discovery; a new fire still
        # has to be able to claim its own detections.
        self.assertEqual(claim_radius_km(0), CLAIM_MIN_KM)
        self.assertEqual(claim_radius_km(None), CLAIM_MIN_KM)

    def test_ceiling_applies_to_megafires(self):
        self.assertEqual(claim_radius_km(5_000_000), CLAIM_MAX_KM)

    def test_monotonic(self):
        rs = [claim_radius_km(a) for a in (0, 10, 100, 1e3, 1e4, 1e5, 1e6)]
        self.assertEqual(rs, sorted(rs))


class TestExclusivity(unittest.TestCase):
    """The core guarantee: no detection is ever counted twice."""

    def test_every_detection_lands_exactly_once(self):
        fires = [{"id": "A", "lat": 39.0, "lon": -120.0, "acres": 5000},
                 {"id": "B", "lat": 39.05, "lon": -120.0, "acres": 5000},
                 {"id": "C", "lat": 39.10, "lon": -120.0, "acres": 5000}]
        dets = [at(39.0 + i * 0.005, -120.0 + j * 0.005)
                for i in range(25) for j in range(5)]
        assigned, unassigned = assign_detections(fires, dets)

        total = sum(len(v) for v in assigned.values()) + len(unassigned)
        self.assertEqual(total, len(dets))

        # and no single detection object appears under two fires
        seen = set()
        for lst in assigned.values():
            for d in lst:
                key = (d["lat"], d["lon"])
                self.assertNotIn(key, seen, "detection assigned twice")
                seen.add(key)

    def test_two_adjacent_fires_do_not_both_claim_the_middle(self):
        # 6 km apart, both 5,000 acres -> claim radius ~4 km each, so their
        # footprints overlap. A radius query would count the middle twice.
        a = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 5000}
        b_lat, b_lon = north_of(39.0, -120.0, 6.0)
        b = {"id": "B", "lat": b_lat, "lon": b_lon, "acres": 5000}
        mid_lat, mid_lon = north_of(39.0, -120.0, 3.0)
        assigned, _ = assign_detections([a, b], [at(mid_lat, mid_lon, 500.0)])
        claimed = [k for k, v in assigned.items() if v]
        self.assertEqual(len(claimed), 1, "both fires claimed the same heat")
        self.assertEqual(sum(d["frp"] for v in assigned.values() for d in v),
                         500.0, "total FRP changed under assignment")

    def test_contested_detection_is_flagged(self):
        a = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 5000}
        b_lat, b_lon = north_of(39.0, -120.0, 6.0)
        b = {"id": "B", "lat": b_lat, "lon": b_lon, "acres": 5000}
        mid_lat, mid_lon = north_of(39.0, -120.0, 3.0)
        assigned, _ = assign_detections([a, b], [at(mid_lat, mid_lon)])
        det = next(d for v in assigned.values() for d in v)
        self.assertTrue(det["_contested"],
                        "a detection halfway between two equal fires "
                        "should be reported as contested")

    def test_detection_clearly_inside_one_fire_is_not_contested(self):
        a = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 5000}
        b = {"id": "B", "lat": 39.9, "lon": -120.0, "acres": 5000}  # ~100 km
        assigned, _ = assign_detections([a, b], [at(39.0, -120.0)])
        det = assigned["A"][0]
        self.assertFalse(det["_contested"])


class TestSizeDisparity(unittest.TestCase):
    """Nearest-centroid gets these wrong; normalised distance gets them right."""

    def test_megafire_beats_nearer_small_fire(self):
        # Rowe Creek Complex scale (373,929 acres -> ~22 km radius) with a new
        # 50-acre start 8 km from its centroid. A detection 4 km from the small
        # fire is NEARER it, but sits 1.3 claim-radii outside its footprint and
        # well inside the complex. Nearest-centroid would hand a 50-acre fire
        # heat that belongs to a 374,000-acre one.
        big = {"id": "BIG", "lat": 44.5, "lon": -120.0, "acres": 373929}
        small_lat, small_lon = north_of(44.5, -120.0, 8.0)
        small = {"id": "SMALL", "lat": small_lat, "lon": small_lon, "acres": 50}
        d_lat, d_lon = north_of(44.5, -120.0, 4.0)

        det = at(d_lat, d_lon, 300.0)
        # confirm the premise: it really is nearer the small fire
        self.assertLess(haversine_km(d_lat, d_lon, small_lat, small_lon),
                        haversine_km(d_lat, d_lon, 44.5, -120.0))

        assigned, _ = assign_detections([big, small], [det])
        self.assertIn("BIG", assigned)
        self.assertNotIn("SMALL", assigned)

    def test_small_fire_still_keeps_its_own_core(self):
        # The guard must not go so far that a small fire loses heat at its
        # own centre to a nearby large one.
        big = {"id": "BIG", "lat": 44.5, "lon": -120.0, "acres": 373929}
        small_lat, small_lon = north_of(44.5, -120.0, 8.0)
        small = {"id": "SMALL", "lat": small_lat, "lon": small_lon, "acres": 50}
        assigned, _ = assign_detections([big, small], [at(small_lat, small_lon)])
        self.assertIn("SMALL", assigned)

    def test_complex_members_at_same_point_do_not_both_claim(self):
        # Complexes list member fires at nearly identical coordinates.
        members = [{"id": f"M{i}", "lat": 40.0 + i * 1e-4, "lon": -121.0,
                    "acres": 1000} for i in range(4)]
        assigned, unassigned = assign_detections(members, [at(40.0, -121.0, 99.0)])
        self.assertEqual(len(unassigned), 0)
        self.assertEqual(sum(len(v) for v in assigned.values()), 1)


class TestUnassigned(unittest.TestCase):
    def test_heat_far_from_any_fire_is_not_credited(self):
        # This is the case the hotspot layer exists for: heat nobody logged.
        # It must not be silently folded into the nearest fire's total.
        fires = [{"id": "A", "lat": 39.0, "lon": -120.0, "acres": 100}]
        far_lat, far_lon = north_of(39.0, -120.0, 60.0)
        assigned, unassigned = assign_detections(fires, [at(far_lat, far_lon)])
        self.assertEqual(assigned, {})
        self.assertEqual(len(unassigned), 1)

    def test_no_fires_at_all(self):
        assigned, unassigned = assign_detections([], [at(39.0, -120.0)])
        self.assertEqual(assigned, {})
        self.assertEqual(len(unassigned), 1)


class TestOverpasses(unittest.TestCase):
    """FRP is a rate. Summing a day of passes reports several fires' worth."""

    def test_two_passes_hours_apart_are_separate(self):
        dets = ([at(39, -120, 100.0, time="0912") for _ in range(3)] +
                [at(39, -120, 200.0, time="2112") for _ in range(3)])
        passes = group_overpasses(dets)
        self.assertEqual(len(passes), 2)

    def test_one_pass_spanning_a_few_minutes_stays_together(self):
        dets = [at(39, -120, 100.0, time=t) for t in ("2110", "2112", "2114")]
        self.assertEqual(len(group_overpasses(dets)), 1)

    def test_different_satellites_are_separate_observations(self):
        dets = [at(39, -120, 100.0, time="2112", sat="N"),
                at(39, -120, 120.0, time="2114", sat="N20")]
        self.assertEqual(len(group_overpasses(dets)), 2)

    def test_summary_reports_latest_pass_not_the_daily_total(self):
        dets = ([at(39, -120, 100.0, time="0912") for _ in range(3)] +   # 300
                [at(39, -120, 50.0, time="2112") for _ in range(3)])     # 150
        s = summarize_fire(dets)
        self.assertEqual(s["frp_mw"], 150.0, "reported the 24 h sum, not the "
                                             "most recent overpass")
        self.assertEqual(s["frp_peak_mw"], 300.0)
        self.assertEqual(s["n_overpasses"], 2)
        self.assertEqual(s["n_det"], 3)

    def test_summary_of_nothing(self):
        self.assertIsNone(summarize_fire([]))


class TestRealWorldGeometry(unittest.TestCase):
    def test_dome_and_a_neighbour(self):
        # Dome, Mariposa CA: 5,667 acres -> 4.7 km claim radius.
        dome = {"id": "DOME", "lat": 37.48, "lon": -119.97, "acres": 5667}
        nb_lat, nb_lon = north_of(37.48, -119.97, 9.0)
        neighbour = {"id": "NB", "lat": nb_lat, "lon": nb_lon, "acres": 300}
        # A cluster on Dome's north flank, 3 km up -- inside Dome, outside NB.
        dets = [at(*north_of(37.48, -119.97, 3.0), frp=80.0) for _ in range(5)]
        assigned, unassigned = assign_detections([dome, neighbour], dets)
        self.assertEqual(len(assigned.get("DOME", [])), 5)
        self.assertEqual(len(assigned.get("NB", [])), 0)
        self.assertEqual(summarize_fire(assigned["DOME"])["frp_mw"], 400.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


def square(lat, lon, half_deg):
    """GeoJSON Polygon ring, [lon, lat] order, closed."""
    return {"type": "Polygon", "coordinates": [[
        [lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
        [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg],
        [lon - half_deg, lat - half_deg]]]}


class TestPerimeterAssignment(unittest.TestCase):
    """A published perimeter is stronger evidence than any circle."""

    def test_perimeter_beats_a_nearer_fire_centroid(self):
        # B's centroid is nearer the detection, but the detection lies inside
        # A's mapped perimeter. The polygon must win: it is measured, the
        # circle is inferred.
        a = {"id": "A", "lat": 39.00, "lon": -120.00, "acres": 40000}
        b = {"id": "B", "lat": 39.04, "lon": -120.00, "acres": 40000}
        perims = {"A": square(39.00, -120.00, 0.05)}
        det = at(39.035, -120.00, 250.0)
        self.assertLess(haversine_km(39.035, -120.0, b["lat"], b["lon"]),
                        haversine_km(39.035, -120.0, a["lat"], a["lon"]))
        assigned, _ = assign_detections([a, b], [det], perims)
        self.assertIn("A", assigned)
        self.assertNotIn("B", assigned)
        self.assertTrue(assigned["A"][0]["_in_perim"])

    def test_detection_in_an_unburned_hole_is_not_inside(self):
        geom = square(39.0, -120.0, 0.10)
        # interior ring: an unburned island at the centre
        geom["coordinates"].append([[-120.02, 38.98], [-119.98, 38.98],
                                    [-119.98, 39.02], [-120.02, 39.02],
                                    [-120.02, 38.98]])
        fire = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 40000}
        assigned, _ = assign_detections([fire], [at(39.0, -120.0)], {"A": geom})
        # it falls back to the claim radius rather than counting as "inside"
        self.assertFalse(assigned["A"][0]["_in_perim"])

    def test_nested_complex_gives_the_member_not_the_complex(self):
        # Complexes publish a perimeter enclosing their members' perimeters.
        # The smallest containing polygon is the specific answer.
        cx = {"id": "CX", "lat": 44.0, "lon": -120.0, "acres": 300000}
        member = {"id": "M", "lat": 44.0, "lon": -120.0, "acres": 9000}
        perims = {"CX": square(44.0, -120.0, 0.50),
                  "M": square(44.0, -120.0, 0.05)}
        assigned, _ = assign_detections([cx, member], [at(44.0, -120.0)], perims)
        self.assertIn("M", assigned)
        self.assertNotIn("CX", assigned)

    def test_perimeter_for_an_unknown_fire_is_ignored(self):
        # A perimeter whose id is not in the current fire list must not create
        # a phantom entry keyed to a fire the page cannot look up.
        fire = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 100}
        perims = {"GHOST": square(39.0, -120.0, 0.20)}
        assigned, _ = assign_detections([fire], [at(39.0, -120.0)], perims)
        self.assertEqual(set(assigned), {"A"})

    def test_exclusivity_still_holds_with_perimeters(self):
        fires = [{"id": f"F{i}", "lat": 39.0 + i * 0.02, "lon": -120.0,
                  "acres": 5000} for i in range(4)]
        perims = {f"F{i}": square(39.0 + i * 0.02, -120.0, 0.03)
                  for i in range(4)}     # deliberately overlapping squares
        dets = [at(39.0 + i * 0.004, -120.0 + j * 0.004)
                for i in range(16) for j in range(4)]
        assigned, unassigned = assign_detections(fires, dets, perims)
        self.assertEqual(sum(len(v) for v in assigned.values()) + len(unassigned),
                         len(dets))
        keys = [(d["lat"], d["lon"]) for v in assigned.values() for d in v]
        self.assertEqual(len(keys), len(set(keys)), "detection counted twice")

    def test_no_perimeters_behaves_exactly_as_before(self):
        fires = [{"id": "A", "lat": 39.0, "lon": -120.0, "acres": 5000}]
        d = [at(39.0, -120.0, 50.0)]
        self.assertEqual(assign_detections(fires, d)[0]["A"][0]["frp"],
                         assign_detections(fires, d, {})[0]["A"][0]["frp"])


class TestSummaryFields(unittest.TestCase):
    def test_total_hottest_and_peak_are_distinguished(self):
        # Forty quiet pixels and one violent head can share a total; the
        # hottest-pixel figure is what tells them apart.
        dets = [at(39, -120, 10.0, time="2112") for _ in range(10)]
        dets.append(at(39, -120, 300.0, time="2112"))
        s = summarize_fire(dets)
        self.assertEqual(s["frp_mw"], 400.0)
        self.assertEqual(s["frp_max_pixel_mw"], 300.0)

    def test_in_perimeter_count_is_reported(self):
        a = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 40000}
        perims = {"A": square(39.0, -120.0, 0.05)}
        dets = [at(39.0, -120.0, 10.0), at(39.001, -120.001, 10.0)]
        assigned, _ = assign_detections([a], dets, perims)
        self.assertEqual(summarize_fire(assigned["A"])["n_in_perim"], 2)


class TestStalePerimeterBuffer(unittest.TestCase):
    """
    A perimeter is a snapshot. Six to eight hours later an active flank has
    moved beyond it, and that heat is still this fire's.
    """

    def test_detection_just_outside_is_still_this_fire(self):
        fire = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 40000}
        perims = {"A": square(39.0, -120.0, 0.05)}     # edge at 39.05
        just_out, lon = north_of(39.05, -120.0, 1.2)   # 1.2 km beyond it
        assigned, unassigned = assign_detections([fire], [at(just_out, lon)], perims)
        self.assertEqual(len(unassigned), 0)
        det = assigned["A"][0]
        self.assertFalse(det["_in_perim"])
        self.assertTrue(det["_near_perim"])

    def test_detection_far_outside_is_not_swept_in(self):
        # The buffer must not become a licence to claim distant heat. 12 km
        # out is beyond both the buffer AND this fire's claim radius.
        fire = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 1000}
        perims = {"A": square(39.0, -120.0, 0.01)}
        far, lon = north_of(39.01, -120.0, 12.0)
        assigned, unassigned = assign_detections([fire], [at(far, lon)], perims)
        self.assertEqual(len(unassigned), 1)

    def test_buffer_cannot_reach_across_to_a_neighbour(self):
        # Two fires 10 km apart, each with a tight perimeter. A detection on
        # A's flank must go to A, never to B.
        a = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 2000}
        b_lat, b_lon = north_of(39.0, -120.0, 10.0)
        b = {"id": "B", "lat": b_lat, "lon": b_lon, "acres": 2000}
        perims = {"A": square(39.0, -120.0, 0.01),
                  "B": square(b_lat, b_lon, 0.01)}
        det_lat, det_lon = north_of(39.01, -120.0, 1.0)   # just off A's edge
        assigned, _ = assign_detections([a, b], [at(det_lat, det_lon)], perims)
        self.assertIn("A", assigned)
        self.assertNotIn("B", assigned)

    def test_nearest_boundary_wins_when_two_buffers_overlap(self):
        # Adjacent fires whose buffers overlap: still exactly one owner.
        a = {"id": "A", "lat": 39.00, "lon": -120.0, "acres": 2000}
        b_lat, b_lon = north_of(39.00, -120.0, 3.0)
        b = {"id": "B", "lat": b_lat, "lon": b_lon, "acres": 2000}
        perims = {"A": square(39.00, -120.0, 0.005),
                  "B": square(b_lat, b_lon, 0.005)}
        # 0.5 km above A's edge. (A first draft used 1.0 km and failed: that
        # put the point 1.0 km from A's boundary but 0.89 km from B's, so the
        # code was right and the fixture was wrong. Worth the arithmetic.)
        det_lat, det_lon = north_of(39.005, -120.0, 0.5)
        assigned, _ = assign_detections([a, b], [at(det_lat, det_lon)], perims)
        self.assertEqual(sum(len(v) for v in assigned.values()), 1)
        self.assertIn("A", assigned)

    def test_inside_still_beats_near(self):
        # Containment is stronger evidence than proximity: a detection inside
        # A must go to A even if it sits nearer B's boundary.
        a = {"id": "A", "lat": 39.0, "lon": -120.0, "acres": 40000}
        b_lat, b_lon = north_of(39.0, -120.0, 6.0)
        b = {"id": "B", "lat": b_lat, "lon": b_lon, "acres": 2000}
        perims = {"A": square(39.0, -120.0, 0.04),
                  "B": square(b_lat, b_lon, 0.005)}
        inside_a_near_b, lon = north_of(39.0, -120.0, 4.2)   # inside A's square
        assigned, _ = assign_detections([a, b], [at(inside_a_near_b, lon)], perims)
        self.assertIn("A", assigned)
        self.assertTrue(assigned["A"][0]["_in_perim"])

    def test_exclusivity_holds_with_buffers(self):
        fires = [{"id": f"F{i}", "lat": 39.0 + i * 0.02, "lon": -120.0,
                  "acres": 5000} for i in range(5)]
        perims = {f"F{i}": square(39.0 + i * 0.02, -120.0, 0.008)
                  for i in range(5)}
        dets = [at(38.99 + i * 0.004, -120.0 + j * 0.004)
                for i in range(30) for j in range(4)]
        assigned, unassigned = assign_detections(fires, dets, perims)
        self.assertEqual(sum(len(v) for v in assigned.values()) + len(unassigned),
                         len(dets))
        keys = [(d["lat"], d["lon"]) for v in assigned.values() for d in v]
        self.assertEqual(len(keys), len(set(keys)), "detection counted twice")

    def test_distance_is_zero_inside_and_grows_outside(self):
        from frp_assign import distance_to_geometry_km
        g = square(39.0, -120.0, 0.05)
        self.assertEqual(distance_to_geometry_km(-120.0, 39.0, g), 0.0)
        lat2, _ = north_of(39.05, -120.0, 2.0)
        self.assertAlmostEqual(distance_to_geometry_km(-120.0, lat2, g), 2.0,
                               delta=0.05)
