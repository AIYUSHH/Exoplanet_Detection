# Exoplanet Detection

This repository contains a Python pipeline for detecting transit-like signals in
Kepler light curves. It is intended for experimenting with automated
exoplanet-candidate detection, signal processing, feature engineering, and
dataset creation for machine-learning models.

## What it does

For each target light curve, the pipeline:

1. Loads time, flux, and quality arrays from a NumPy `.npz` file.
2. Removes poor-quality and non-finite observations.
3. Normalizes and detrends the flux to isolate short transit signals.
4. Searches a grid of trial orbital periods using a Box Least Squares (BLS)
   period search.
5. Selects the strongest candidate and measures its period, epoch, duration,
   depth, signal residue (SR), and signal detection efficiency (SDE).
6. Extracts transit-shape, noise, periodicity, and symmetry features.
7. Saves a summary table and diagnostic periodogram and folded-light-curve
   plots.

The project also includes helpers for loading the NASA Kepler Object of
Interest (KOI) catalog, downloading Kepler light curves with Lightkurve, and
building labeled datasets from confirmed planets and false positives.

## Requirements

- Python 3.11 or newer
- Internet access for downloading catalog data or Kepler light curves
- The dependencies listed in `pyproject.toml` or `requirements.txt`

## Installation

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

## Input data

Processing scripts expect light curves in `data/raw/`. Each `.npz` file must
contain these arrays:

```text
time      observation times
flux      measured stellar flux
quality   Kepler quality flags
```

The repository ignores `data/raw/` because downloaded light curves can be
large. The catalog is stored in `data/catalogs/koi_catalog.csv`, and target
lists are stored in `data/targets/`.

## How to run

### Download Kepler light curves

`download_stage1.py` loads the local KOI catalog and downloads one long-cadence
Kepler light curve per unique KEPID:

```bash
uv run python download_stage1.py
```

Downloaded files are written to `data/raw/`.

### Process all downloaded targets

`run_dataset.py` processes every `.npz` file in `data/raw/`:

```bash
uv run python run_dataset.py
```

The batch pipeline preprocesses each target, runs BLS, and writes:

```text
outputs/summary.csv
outputs/periodograms/<target>.png
outputs/folded_curves/<target>.png
```

### Run the single-target validation pipeline

`main.py` demonstrates the complete profiling and plotting workflow for one
target. Set `FILE` near the top of the file to the path of an available `.npz`
light curve before running it:

```bash
uv run python main.py
```

This prints timing, memory, candidate, and feature information and saves
`plots/validation_result.png`.

### Build a labeled dataset

`build_data_stage1.py` selects a small sample of confirmed planets and false
positives, runs the detection pipeline, and writes a dataset:

```bash
uv run python build_data_stage1.py
```

The generated dataset is saved under `datasets/`, which is ignored by Git.
The more configurable functions for dataset construction are implemented in
`dataset_builder.py`.

### Validate recovered periods and features

After the required catalog and raw light curves are available:

```bash
uv run python validation.py
```

Validation results are written to `validation_results.csv`.

## Project layout

```text
.
├── data/
│   ├── catalogs/       KOI catalog data
│   └── targets/        target lists
├── src/
│   ├── downloader.py   Kepler download helpers
│   ├── preprocessing.py
│   ├── bls_numba.py    accelerated BLS search
│   ├── features.py     transit and noise features
│   ├── batch.py        multi-target processing
│   └── plotting.py      diagnostic plots
├── download_stage1.py  download catalog targets
├── run_dataset.py      process all raw light curves
├── build_data_stage1.py
├── main.py              single-target profiling example
└── validation.py        pipeline validation
```

## Important notes

- This project identifies transit-like signals; a detected signal is not by
  itself proof that a planet exists.
- The default period grid searches periods from approximately 0.5 to 30 days.
- Downloaded data, generated datasets, plots, caches, and virtual environments
  are excluded by `.gitignore`.