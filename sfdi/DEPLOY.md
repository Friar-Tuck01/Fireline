# Deploying the daily SFDI job

The workflow in `.github/workflows/sfdi-daily.yml` runs every morning, but it
needs a one-time setup first. It will fail with a clear error until you do
this, rather than publishing anything wrong.

## Why there is setup at all

ERC is a *build-up* index. The 1000-hour fuels carry weeks of memory and the
live fuel models track a damped version of that, so today's value cannot be
computed from today's weather — the fuel-moisture state has to be carried
forward, unbroken, from the climatology.

That state (`state.pkl`, ~40 MB) is the one artifact here that cannot be
regenerated quickly. Rebuilding it means re-running 1979 onward, which is
hours of downloading. So it has to persist between Action runs, and it has to
persist somewhere that is not git history — a 40 MB file changing daily would
add roughly 14 GB a year to the repo.

A **release asset** is the right home: replaceable in place, outside the git
tree, no history accumulation.

## Step 1 — commit the static grid files

These define the grid and never change. From your local work directory:

    ~/fireline-sfdi-work/meta/climate_class.npy
    ~/fireline-sfdi-work/meta/lat.npy
    ~/fireline-sfdi-work/meta/lons.npy

Upload all three to `data/` in the repo (about 2 MB total). `data/` should
then hold:

    climate_class.npy      grid + NFDRS climate class per cell
    lat.npy                latitude of every grid row
    lons.npy               longitude of every grid column
    sfdi_lookup.npz        the 1979-2017 climatology lookup tables
    sfdi_latest.png        today's map (the Action overwrites this)
    sfdi_latest.json       manifest (the Action overwrites this)

## Step 2 — create the state release

On GitHub: **Releases → Draft a new release**.

- Tag: `sfdi-state` (must match `STATE_TAG` in the workflow)
- Title: `SFDI fuel-moisture state`
- Description: something like *"Rolling NFDRS state. Replaced daily by the
  SFDI workflow. Not a software release."*
- Attach these two files from `~/fireline-sfdi-work/meta/`:
  - `state.pkl`
  - `last_date.json`
- Check **Set as a pre-release** so it does not read as a version of Fireline.

Publish it.

## Step 3 — run it by hand once

**Actions → SFDI daily → Run workflow.** Watch it. A good run:

1. downloads `state.pkl` from the release
2. fetches a few days of gridMET through THREDDS
3. steps the engine forward and renders the PNG
4. sanity-checks the output before publishing anything
5. uploads the new state, then commits the map

The order in steps 4-5 is deliberate. State is saved *before* the map is
committed, so a failure leaves the published map matching the stored state.
The reverse could leave a map on the site that the state can never reproduce.

## What "no change" means

gridMET observations lag 2-4 days. On mornings when nothing new has been
published, the job prints `state is already current`, skips the commit, and
exits successfully. That is normal, not a failure.

## If the state is ever lost

Not fatal, just slow. Rebuild locally:

    python3 build_climatology.py prepass
    caffeinate -i python3 build_climatology.py run     # ~1 hr
    python3 build_breakpoints.py
    caffeinate -i python3 catchup.py                   # ~20 min
    python3 daily_sfdi.py run

then re-upload `state.pkl` to the `sfdi-state` release. The climatology year
files in `~/fireline-sfdi-work/erc_bi/` (13.8 GB) are worth keeping precisely
so this stays a rerun rather than a re-download.

## Guards worth knowing about

- **`MAX_CATCHUP_DAYS = 30`** in `daily_sfdi.py`. If the job has been failing
  quietly for a month, it stops rather than grinding through a season's
  backlog. Raise it deliberately, not reflexively.
- **Output validation** in the workflow refuses to publish a PNG under 5 KB or
  a field that is more than 98% Low. Better to leave yesterday's map up than
  replace a good one with a broken one.
- **`concurrency`** prevents overlapping runs. Two runs starting from the same
  state would silently lose a day.
