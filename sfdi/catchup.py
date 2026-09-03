"""
Carry the NFDRS state forward from the end of the climatology to now.

The climatology stops at 2017-12-31 and `state.pkl` holds the fuel-moisture
state as of that date. ERC is a build-up index -- MC1000 has weeks of memory
and the live fuel models track a damped version of it -- so today's value
cannot be computed from today's weather alone. The state has to be walked
forward, day by day, from where the climatology left it.

This is the bridge between the one-time climatology and the daily product.
Run it once; after that the daily job steps the state forward a few days at
a time.

    python3 catchup.py            # 2018 -> latest complete year
    python3 catchup.py status

Output beyond the updated state: a rolling buffer of recent ERC and BI, kept
so the daily product can render a short history and so we can spot-check
recent fire seasons against the climatology.

NOTE ON YEARS. gridMET's current-year file grows through the year. This
script handles COMPLETE years only; picking up the partial current year is
the daily job's business, and it uses a different fetch path (THREDDS
subsetting for a few days rather than a 350 MB annual file).
"""

import datetime as dt
import os
import pickle
import sys

import numpy as np

import build_climatology as bc

# Keep this many recent days of ERC/BI for the daily product and for spot
# checks. Two years is enough to show a full season either side of today.
BUFFER_DAYS = 730

RECENT = os.path.join(bc.WORK, "recent")


def _state_path():
    return os.path.join(bc.META, "state.pkl")


def status():
    sp = _state_path()
    if not os.path.exists(sp):
        sys.exit("no state.pkl -- run build_climatology first")
    with open(sp, "rb") as f:
        _, next_year = pickle.load(f)
    print(f"state is current through {next_year - 1}-12-31")
    print(f"next year to process: {next_year}")
    latest = dt.date.today().year - 1
    if next_year > latest:
        print("state is up to date through the last complete year")
    else:
        print(f"catch-up needed: {next_year}..{latest} "
              f"({latest - next_year + 1} years, "
              f"~{2 * (latest - next_year + 1)} min)")
    if os.path.exists(RECENT):
        for f in sorted(os.listdir(RECENT)):
            p = os.path.join(RECENT, f)
            print(f"  {f}: {os.path.getsize(p)/1e6:.1f} MB")


def main():
    os.makedirs(RECENT, exist_ok=True)
    sp = _state_path()
    if not os.path.exists(sp):
        sys.exit("no state.pkl -- run build_climatology first")
    with open(sp, "rb") as f:
        st, next_year = pickle.load(f)

    latest = dt.date.today().year - 1
    if next_year > latest:
        print(f"already current through {next_year - 1}; nothing to do")
        return

    cc = np.load(os.path.join(bc.META, "climate_class.npy"))
    shape2d = cc.shape
    lat2d = np.load(os.path.join(bc.META, "lat.npy"))
    lat_flat = lat2d.ravel().astype(np.float64)

    print(f"carrying state {next_year}..{latest}")
    for year in range(next_year, latest + 1):
        t0 = dt.datetime.now()
        # Reuse the climatology's own year processor so the catch-up and the
        # climatology cannot drift apart. Same code, same assumptions, same
        # NaN handling -- which is the whole reason SFDI percentiles mean
        # anything.
        bc._process_year(st, year, lat_flat, shape2d, write=True)

        # _process_year writes into the climatology directory; move the
        # post-2017 years out so they are never mistaken for climatology.
        for kind in ("erc", "bi"):
            src = os.path.join(bc.OUT, f"{kind}_{year}.npy")
            dst = os.path.join(RECENT, f"{kind}_{year}.npy")
            if os.path.exists(src):
                os.replace(src, dst)

        for v in bc.MET_VARS:
            p = os.path.join(bc.RAW, f"{v}_{year}.nc")
            if os.path.exists(p):
                os.remove(p)

        with open(sp, "wb") as f:
            pickle.dump((st, year + 1), f)
        print(f"  {year} done in "
              f"{(dt.datetime.now()-t0).total_seconds()/60:.1f} min")

    _trim_buffer()
    print(f"\nstate now current through {latest}-12-31")
    print("next: the daily job picks up the partial current year")


def _trim_buffer():
    """Keep only the most recent BUFFER_DAYS in the rolling buffer."""
    files = sorted(f for f in os.listdir(RECENT) if f.startswith("erc_"))
    if not files:
        return
    years = [int(f[4:8]) for f in files]
    total = 0
    keep = []
    for y in sorted(years, reverse=True):
        a = np.load(os.path.join(RECENT, f"erc_{y}.npy"), mmap_mode="r")
        total += a.shape[0]
        keep.append(y)
        if total >= BUFFER_DAYS:
            break
    for y in years:
        if y not in keep:
            for kind in ("erc", "bi"):
                p = os.path.join(RECENT, f"{kind}_{y}.npy")
                if os.path.exists(p):
                    os.remove(p)
                    print(f"  trimmed {kind}_{y}.npy")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": main, "status": status}[cmd]()
