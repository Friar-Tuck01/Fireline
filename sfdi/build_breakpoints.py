"""
Turn the 39-year ERC/BI climatology into per-cell SFDI lookup tables.

THE FORMULA (Jolly et al. 2019, Fire 2(3):47):

    p  = ERC' x BI'          product of the two percentiles
    p' = (ERC' x BI')'       percentile of that product
    SFDI = class of p' at the 60/80/90/97 breakpoints

So SFDI is a percentile of a product of percentiles -- three distributions,
not one. To classify a single day we need, for each grid cell:

    1. the climatological distribution of ERC   (today's ERC  -> ERC')
    2. the climatological distribution of BI    (today's BI   -> BI')
    3. the climatological distribution of p     (today's p    -> p')

This script builds all three from the year files and writes them once.

OUTPUT
    erc_lut  uint8  [ncell, 128]  percentile for each integer ERC value
    bi_lut   uint8  [ncell, 256]  percentile for each integer BI value
    p_thresh float32[ncell, 4]    p at the 60/80/90/97 percentiles

ERC and BI come out of the engine as integers, so a direct value-to-percentile
lookup is exact and compact -- no interpolation, no stored quantile curve.
NCAR's implementation adds uniform jitter to break ties among identical
values; we use midranks instead, which is deterministic and needs no RNG.

The tables stay in the repo for the daily Action to use. They are NOT shipped
to the browser -- the Action does the classification and publishes a PNG.

    python3 build_breakpoints.py
"""

import os
import sys

import numpy as np

import build_climatology as bc

ERC_MAX = 128     # engine ceiling is ~100; headroom is cheap at uint8
BI_MAX = 256
TILE = 5000       # cells per pass -- see the memory note below

# MEMORY. Each tile holds ndays x TILE arrays and np.percentile makes a
# float64 copy internally, so the tile size sets peak RAM. At 14,245 days:
#   TILE=20000 -> ~5 GB peak (float64 copy alone is 2.3 GB)
#   TILE=5000  -> ~1.2 GB peak
# The product p is kept as uint16 rather than float32 for the same reason:
# percentiles top out at 100, so 100*100 = 10,000 fits comfortably and the
# array is a quarter the size of the float32 version.


def midrank_percentile(values_by_day, vmax):
    """Percentile of every integer value 0..vmax-1, using midranks.

    Midrank means a value's percentile is the fraction below it plus half the
    fraction equal to it. With integer indices there are many ties, and taking
    the lower edge would push a whole tied block below a class boundary
    together. Midranks split the block, which is what the jitter in NCAR's
    version accomplishes stochastically.

    values_by_day: (ndays, ncells) int16
    returns:       (ncells, vmax) uint8
    """
    ndays, ncells = values_by_day.shape
    out = np.zeros((ncells, vmax), dtype=np.uint8)
    # Histogram per cell, then a cumulative sum, is far cheaper than sorting
    # 14,000 days per cell.
    counts = np.zeros((ncells, vmax), dtype=np.int32)
    v = np.clip(values_by_day, 0, vmax - 1).astype(np.int32)
    for d in range(ndays):
        np.add.at(counts, (np.arange(ncells), v[d]), 1)
    below = np.cumsum(counts, axis=1) - counts        # strictly less than
    pct = (below + 0.5 * counts) / float(ndays) * 100.0
    np.clip(np.rint(pct), 0, 100, out=out, casting="unsafe")
    return out


def main():
    out_dir = os.path.join(bc.WORK, "breakpoints")
    os.makedirs(out_dir, exist_ok=True)

    years = sorted(int(f[4:8]) for f in os.listdir(bc.OUT)
                   if f.startswith("erc_") and f.endswith(".npy"))
    if not years:
        sys.exit(f"no year files in {bc.OUT} -- run build_climatology first")
    print(f"years: {years[0]}-{years[-1]} ({len(years)} files)")

    cc = np.load(os.path.join(bc.META, "climate_class.npy"))
    shape2d = cc.shape
    ncell = cc.size
    land = np.isfinite(cc).ravel()
    print(f"grid {shape2d} = {ncell:,} cells, {int(land.sum()):,} land")

    # Memory-map so we can slice by cell without loading 14 GB.
    erc_mm = [np.load(os.path.join(bc.OUT, f"erc_{y}.npy"), mmap_mode="r")
              for y in years]
    bi_mm = [np.load(os.path.join(bc.OUT, f"bi_{y}.npy"), mmap_mode="r")
             for y in years]
    ndays = sum(a.shape[0] for a in erc_mm)
    print(f"days: {ndays:,}\n")

    erc_lut = np.zeros((ncell, ERC_MAX), dtype=np.uint8)
    bi_lut = np.zeros((ncell, BI_MAX), dtype=np.uint8)
    p_thresh = np.zeros((ncell, 4), dtype=np.float32)

    for start in range(0, ncell, TILE):
        stop = min(start + TILE, ncell)
        erc = np.concatenate([a[:, start:stop] for a in erc_mm], axis=0)
        bi = np.concatenate([a[:, start:stop] for a in bi_mm], axis=0)

        el = midrank_percentile(erc, ERC_MAX)
        bl = midrank_percentile(bi, BI_MAX)
        erc_lut[start:stop] = el
        bi_lut[start:stop] = bl

        # Rebuild the percentile series through the same lookup the daily
        # product will use. Deriving p from the LUT rather than from a
        # separate ranking guarantees the climatology and the live product
        # agree exactly -- the same discipline that made us use one engine
        # for both ends of this.
        cells = np.arange(stop - start)
        ercp = el[cells[None, :], np.clip(erc, 0, ERC_MAX - 1)]
        bip = bl[cells[None, :], np.clip(bi, 0, BI_MAX - 1)]
        # uint16: both percentiles are 0-100, so the product maxes at 10,000.
        p = ercp.astype(np.uint16) * bip.astype(np.uint16)
        del ercp, bip, erc, bi

        p_thresh[start:stop] = np.percentile(
            p, [60, 80, 90, 97], axis=0).T.astype(np.float32)
        p_median = float(np.median(p))   # grab before freeing
        del p

        print(f"  cells {start:>7,}-{stop:>7,}  "
              f"p median {p_median:.0f}  "
              f"p97 {np.mean(p_thresh[start:stop, 3]):.0f}")

    np.savez_compressed(
        os.path.join(out_dir, "sfdi_lookup.npz"),
        erc_lut=erc_lut, bi_lut=bi_lut, p_thresh=p_thresh,
        shape=np.array(shape2d), years=np.array([years[0], years[-1]]))

    path = os.path.join(out_dir, "sfdi_lookup.npz")
    print(f"\nwrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    # --- sanity: the climatology must reproduce its own definition ---------
    # By construction each cell should spend 60/20/10/7/3 percent of days in
    # the five classes. If it does not, the percentile machinery is wrong.
    print("\nverifying class frequencies on a sample of cells")
    idx = np.where(land)[0][::5000][:40]
    fracs = []
    for c in idx:
        erc = np.concatenate([a[:, c] for a in erc_mm])
        bi = np.concatenate([a[:, c] for a in bi_mm])
        ep = erc_lut[c][np.clip(erc, 0, ERC_MAX - 1)].astype(np.float32)
        bp = bi_lut[c][np.clip(bi, 0, BI_MAX - 1)].astype(np.float32)
        p = ep * bp
        t = p_thresh[c]
        cls = np.digitize(p, t)          # 0..4
        fracs.append([np.mean(cls == k) for k in range(5)])
    fracs = np.array(fracs).mean(axis=0) * 100
    names = ["Low", "Moderate", "High", "Very High", "Severe"]
    target = [60, 20, 10, 7, 3]
    print(f"{'class':<11}{'actual':>9}{'target':>9}")
    ok = True
    for nm, a, t in zip(names, fracs, target):
        flag = "" if abs(a - t) < 2.5 else "   <-- off"
        if flag:
            ok = False
        print(f"{nm:<11}{a:>8.1f}%{t:>8}%{flag}")
    print("\n" + ("class frequencies match the definition -- lookups are sound"
                  if ok else
                  "MISMATCH: the percentile tables do not reproduce the\n"
                  "climatology they were built from. Do not build on this."))


if __name__ == "__main__":
    main()
