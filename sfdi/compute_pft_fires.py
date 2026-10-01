"""
compute_pft_fires.py -- per-fire PyroCb Firepower Threshold for Fireline.

Pulls the current WFIGS active-fire list, extracts an HRRR vertical profile at
each fire, runs the Tory & Kepert (2021) PFT solver, and writes
data/pft_latest.json for index.html to read.

    conda activate fireline
    python compute_pft_fires.py --out ../data/pft_latest.json

----------------------------------------------------------------------------
What PFT is, restated here because the sign is counterintuitive
----------------------------------------------------------------------------
PFT is the firepower, in gigawatts, that a fire WOULD NEED for the atmosphere
above it to support deep pyroconvection.

    LOW PFT  = the atmosphere is MORE conducive to a pyroCb
    HIGH PFT = the atmosphere is LESS conducive

Any colour ramp must run low = conducive. It is a threshold the atmosphere
sets, not a measure of how dangerous the fire is, and it says nothing about
whether a given fire will actually produce a pyroCb.

----------------------------------------------------------------------------
Accuracy, honestly
----------------------------------------------------------------------------
Validated against the four archived soundings the paper analysed by hand
(validate_soundings.py). Three of four land within 1.5x of the published PFT,
with z_fc within 3-12% and U_ML within 1-13%.

The fourth -- Chisholm 0600 LST, a morning sounding under a nocturnal
inversion -- comes out 2.15x high, because the step-1 iteration deepens the
mixed layer past the real one (4400 m against a near-isentropic layer that the
paper puts at ~650 hPa) and over-dries q_ML. This is why the run targets 21Z
and why `ml_deep` is flagged per fire. Treat a flagged value as low confidence.

The authors expect "both biases and random errors" and recommend comparing
values RELATIVELY. Fire A against fire B on the same run, or this run against
the last, is a much sounder use than any single absolute number.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

import pft
from pft import Profile, compute_pft
import hrrr_pft

# Same service and domain index.html uses, so the ids line up.
POINTS_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/"
              "services/WFIGS_Incident_Locations_Current/FeatureServer/0/query")
WESTERN_BBOX = {"xmin": -125, "ymin": 31, "xmax": -102, "ymax": 49.5}

USER_AGENT = "fireline-pft/1.0 (+https://github.com/Friar-Tuck01/Fireline)"


def fetch_fires(include_rx=False, timeout=60):
    """
    Current WFIGS incident locations in the western domain.

    The id is built exactly as index.html builds it -- UniqueFireIdentifier
    with a name+lat+lon fallback -- so the JSON this writes can be looked up
    directly by the fire the user clicked. If these two ever diverge the popup
    silently shows nothing, so they must stay in step.
    """
    cats = "'WF','CX','RX'" if include_rx else "'WF','CX'"
    params = {
        "where": f"IncidentTypeCategory IN ({cats})",
        "outFields": ("IncidentName,IncidentSize,POOState,POOCounty,"
                      "IncidentTypeCategory,InitialLatitude,InitialLongitude,"
                      "UniqueFireIdentifier"),
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
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or []
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
            "state": (p.get("POOState") or "").replace("US-", ""),
            "category": p.get("IncidentTypeCategory"),
        })
    return fires


def load_terrain(path):
    """
    Optional high-resolution terrain from build_terrain.py (terrain_450m.npz).

    HRRR's orography is a ~3 km average, which in steep country can sit
    hundreds of metres off the actual fire ground. Since z_fc is measured above
    the fire and then SQUARED, that error is worth a factor of 2-4 in PFT on a
    mountain fire. If the file is present its elevation is used instead, and
    either way the source is recorded per fire.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        z = np.load(path)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not read terrain {path}: {e}")
        return None
    keys = set(z.files)
    elev_key = next((k for k in ("elevation", "elev", "z", "dem") if k in keys), None)
    lat_key = next((k for k in ("lat", "lats", "latitude") if k in keys), None)
    lon_key = next((k for k in ("lon", "lons", "longitude") if k in keys), None)
    if not (elev_key and lat_key and lon_key):
        print(f"  WARNING: {path} has keys {sorted(keys)}; expected elevation "
              f"plus lat/lon. Falling back to HRRR orography.")
        return None
    return {"elev": z[elev_key], "lat": z[lat_key], "lon": z[lon_key]}


def terrain_elev(terr, lat, lon):
    """Nearest-neighbour lookup in the high-res terrain grid."""
    if terr is None:
        return None
    la, lo = terr["lat"], terr["lon"]
    if la.ndim == 1 and lo.ndim == 1:
        iy = int(np.abs(la - lat).argmin())
        ix = int(np.abs(lo - lon).argmin())
        val = terr["elev"][iy, ix]
    else:
        d = (la - lat) ** 2 + (lo - lon) ** 2
        iy, ix = np.unravel_index(int(d.argmin()), d.shape)
        val = terr["elev"][iy, ix]
    return None if not np.isfinite(val) else float(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/pft_latest.json",
                    help="output JSON path (default: data/pft_latest.json)")
    ap.add_argument("--valid-hour", type=int, default=21,
                    help="target valid hour UTC (default 21Z ~ peak fire weather)")
    ap.add_argument("--terrain", default="",
                    help="optional terrain_450m.npz for true fire-ground elevation")
    ap.add_argument("--workdir", default=".", help="scratch dir for the GRIB subset")
    ap.add_argument("--limit", type=int, default=0, help="only the N largest fires")
    ap.add_argument("--include-rx", action="store_true",
                    help="include prescribed fires (default: wildfires and complexes only)")
    ap.add_argument("--keep-grib", action="store_true", help="keep the GRIB subset")
    ap.add_argument("--min-acres", type=float, default=0.0,
                    help="skip fires smaller than this")
    args = ap.parse_args()

    print("Fireline PFT -- per-fire PyroCb Firepower Threshold")
    print("  LOW PFT = atmosphere MORE conducive to pyroCb\n")

    print("Fetching active fires from WFIGS ...")
    fires = fetch_fires(include_rx=args.include_rx)
    if args.min_acres > 0:
        fires = [f for f in fires if (f["acres"] or 0) >= args.min_acres]
    fires.sort(key=lambda f: -(f["acres"] or 0))
    if args.limit:
        fires = fires[: args.limit]
    print(f"  {len(fires)} fires in the western domain")
    if not fires:
        print("  nothing to do")
        return 1

    print("Locating an HRRR cycle ...")
    fields, lats, lons, meta = hrrr_pft.fetch_hrrr_subset(
        workdir=args.workdir, target_valid_hour_utc=args.valid_hour,
        keep=args.keep_grib)
    print(f"  using {meta['cycle_utc']} f{meta['forecast_hour']:02d}, "
          f"valid {meta['valid_utc']}")

    terr = load_terrain(args.terrain)
    print(f"  fire-ground elevation from: "
          f"{'terrain_450m.npz' if terr else 'HRRR orography'}")

    print("Indexing the HRRR grid ...")
    tree, shape = hrrr_pft.build_index(lats, lons)
    iy, ix = hrrr_pft.nearest_indices(tree, shape,
                                      [f["lat"] for f in fires],
                                      [f["lon"] for f in fires])

    out_fires, n_ok, n_fail = {}, 0, 0
    print(f"Computing PFT for {len(fires)} fires ...")
    for k, f in enumerate(fires):
        # Deliberately lean. index.html already has name, lat, lon, acres and
        # state from its own WFIGS fetch, so repeating them here would roughly
        # double a file that gets committed four times a day -- ~200 MB of git
        # history a year, the same trap the handoff doc flags for wsa_latest.png.
        # Only `name` is kept, so the file is readable by a human debugging a
        # join failure.
        rec = {"name": f["name"]}
        prof_d, orog = hrrr_pft.profile_at(fields, int(iy[k]), int(ix[k]))
        if prof_d is None:
            rec.update(status="no_profile",
                       note="too few HRRR levels above ground at this point")
            out_fires[f["id"]] = rec
            n_fail += 1
            continue

        # Fire-ground elevation. Prefer real terrain; fall back to HRRR's.
        elev = terrain_elev(terr, f["lat"], f["lon"])
        elev_src = "terrain_450m" if elev is not None else "hrrr_orography"
        if elev is None:
            elev = orog
        # Guard against a terrain value that sits below the profile's own base;
        # z_fc would then include air the profile never described.
        if elev < prof_d["z_m"][0] - 500.0:
            elev = orog
            elev_src = "hrrr_orography_clamped"

        try:
            prof = Profile(prof_d["p_hpa"], prof_d["z_m"], prof_d["T_k"],
                           prof_d["q_gkg"], prof_d["u_ms"], prof_d["v_ms"])
            res = compute_pft(prof, fire_elev_m=elev)
        except Exception as e:  # noqa: BLE001
            # Two legitimate physical outcomes, not crashes, and they mean
            # OPPOSITE things -- so they must not share a status:
            #   too_stable  - the column never reaches -20 C, so no firepower
            #                 produces deep pyroconvection. Least conducive.
            #   degenerate  - the column is already buoyant with no fire at
            #                 all, so PFT is identically zero and the concept
            #                 does not apply. Most conducive, and dangerous to
            #                 render as "0 GW".
            msg = str(e)
            st = ("degenerate" if msg.startswith("Degenerate")
                  else "too_stable" if "-20" in msg or "never reaches" in msg
                  else "no_solution")
            rec.update(status=st, note=msg[:160],
                       fire_elev_m=round(elev, 1), elevation_source=elev_src)
            out_fires[f["id"]] = rec
            n_fail += 1
            continue

        # Potential-temperature spread across the diagnosed mixed layer.
        # Computed here from the profile rather than inside pft.py, so the
        # solver's API stays unchanged.
        zsfc = float(prof.z_m[0])
        in_ml = prof.z_m <= zsfc + res.ml_depth_m
        if in_ml.sum() >= 2:
            th = prof.T_k[in_ml] * (1000.0 / prof.p_hpa[in_ml]) ** pft.KAPPA
            ml_theta_range = float(th.max() - th.min())
        else:
            ml_theta_range = 0.0

        rec.update(
            status="ok",
            pft_gw=round(res.pft_gw, 1),
            z_fc_km=round(res.z_fc_km, 3),
            dtheta_fc_k=round(res.dtheta_fc_k, 2),
            u_ml_ms=round(res.u_ml_ms, 2),
            theta_ml_k=round(res.theta_ml_k, 1),
            q_ml_gkg=round(res.q_ml_gkg, 2),
            ml_depth_m=round(res.ml_depth_m, 0),
            # ml_converged means the ML-LCL sits inside the mixed layer. After
            # the theta cap that is False on nearly every dry western profile
            # and on all four validation soundings, so it is a physical
            # statement, NOT a confidence flag. Recorded, never surfaced as a
            # warning.
            ml_converged=bool(res.ml_converged),
            # The real thinness signal: how many profile levels the ML average
            # used. One or two levels is a weak basis for theta_ML and q_ML.
            ml_levels=int(res.ml_levels),
            ml_thin=bool(res.ml_levels <= 2),
            # How well mixed is the "mixed layer", really?
            #
            # The first cut flagged ml_deep at >= 4 km, calibrated from the one
            # validation case that ran high (Chisholm 0600, a 4400 m ML). On
            # real western afternoon profiles that fires on ~60% of fires,
            # because 4-5 km dry convective boundary layers are NORMAL at 21Z
            # -- the paper itself puts Black Saturday's afternoon ML near 5 km.
            # A flag that trips on the majority carries no information.
            #
            # Chisholm's actual problem was not depth but that the layer was
            # not mixed: the average ran up through stable air above the
            # residual layer, drying q_ML and pushing z_fc up. That is directly
            # measurable as the spread of potential temperature across the
            # layer. A genuinely mixed layer holds theta nearly constant.
            #
            # The threshold below is PROVISIONAL. ml_theta_range_k is recorded
            # on every fire so it can be set from the measured distribution
            # rather than from intuition.
            ml_theta_range_k=round(ml_theta_range, 2),
            ml_not_mixed=bool(ml_theta_range > 3.0),
            fire_elev_m=round(elev, 1),
            hrrr_orography_m=round(orog, 1),
            elevation_source=elev_src,
            levels_used=int(len(prof_d["p_hpa"])),
        )
        out_fires[f["id"]] = rec
        n_ok += 1
        if (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(fires)}", flush=True)

    fires_all = out_fires
    vals = [r["pft_gw"] for r in out_fires.values() if r.get("status") == "ok"]
    payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "Tory & Kepert 2021 simple PFT (Eq. 31), Fireline implementation",
        "reference": "https://doi.org/10.1175/WAF-D-20-0027.1",
        "units": "GW",
        "interpretation": "LOW PFT = atmosphere more conducive to pyroCb",
        "anchors_gw": {
            "sir_ivan_2017": pft.SIR_IVAN_GW,
            "chisholm_afternoon": 100.0,
            "black_saturday_morning": 1240.0,
        },
        "elevation_source": "terrain_450m" if terr else "hrrr_orography",
        "n_fires": len(fires), "n_ok": n_ok, "n_failed": n_fail,
        "pft_percentiles_gw": (
            {p: round(float(np.percentile(vals, p)), 1)
             for p in (10, 25, 50, 75, 90)} if vals else {}),
        "fires": out_fires,
        **meta,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    size = os.path.getsize(args.out)
    print(f"\nWrote {args.out}  ({size / 1024:.1f} KB)")
    print(f"  {n_ok} ok, {n_fail} without a value")
    if vals:
        q = payload["pft_percentiles_gw"]
        print(f"  PFT GW  p10={q[10]}  p25={q[25]}  median={q[50]}  "
              f"p75={q[75]}  p90={q[90]}")
        print(f"  lowest (most conducive): ", end="")
        best = sorted(((r["pft_gw"], r["name"]) for r in out_fires.values()
                       if r.get("status") == "ok"))[:3]
        print(", ".join(f"{n} {v:.0f} GW" for v, n in best))
        nm = sum(1 for r in out_fires.values() if r.get("ml_not_mixed"))
        ranges = sorted(r["ml_theta_range_k"] for r in out_fires.values()
                        if "ml_theta_range_k" in r)
        if ranges:
            mid = ranges[len(ranges) // 2]
            print(f"  ML theta spread (K): min={ranges[0]:.2f} "
                  f"median={mid:.2f} max={ranges[-1]:.2f}")
        print(f"  {nm} fire(s) flagged ml_not_mixed (theta spread > 3 K, "
              f"provisional threshold)")
        thin = sum(1 for r in out_fires.values() if r.get("ml_thin"))
        lv = sorted(r["ml_levels"] for r in out_fires.values() if "ml_levels" in r)
        if lv:
            print(f"  ML levels used: min={lv[0]} median={lv[len(lv)//2]} "
                  f"max={lv[-1]}")
        print(f"  {thin} fire(s) flagged ml_thin (ML average over <= 2 levels)")
        for st in ("degenerate", "too_stable"):
            n = sum(1 for r in fires_all.values() if r.get("status") == st)
            if n:
                print(f"  {n} fire(s) status={st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
