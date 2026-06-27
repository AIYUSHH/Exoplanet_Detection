from time import perf_counter
import numpy as np

from src.loader import load_kepler_data
from src.preprocessing import preprocess
from src.bls_numba import search_period
from src.features import compute_features


# -----------------------------
# CHANGE THIS TO ANY STAR
# -----------------------------
filepath = "data/raw/K00752.01.npz"

period_grid = np.linspace(
    0.5,
    30.0,
    5000
)

nbins = 200
min_duration = 0.01
max_duration = 0.10

print("=" * 60)
print("PIPELINE PROFILER")
print("=" * 60)

# -----------------------------
# LOAD
# -----------------------------
t0 = perf_counter()

time, flux, quality = load_kepler_data(filepath)

t1 = perf_counter()

print(f"Load time           : {t1-t0:.4f} s")

# -----------------------------
# PREPROCESS
# -----------------------------
proc = preprocess(
    time,
    flux,
    quality,
    window=51,
    poly=3
)

clean_time = proc["time"]
clean_flux = proc["flux"]

t2 = perf_counter()

print(f"Preprocess time     : {t2-t1:.4f} s")

# -----------------------------
# BLS
# -----------------------------
result = search_period(
    time=clean_time,
    flux=clean_flux,
    period_grid=period_grid,
    nbins=nbins,
    min_duration=min_duration,
    max_duration=max_duration
)

candidate = result["best"]

t3 = perf_counter()

print(f"BLS time            : {t3-t2:.4f} s")

# -----------------------------
# FEATURES
# -----------------------------
features = compute_features(
    candidate,
    clean_time,
    clean_flux
)

t4 = perf_counter()

print(f"Feature time        : {t4-t3:.4f} s")

# -----------------------------
# TOTAL
# -----------------------------
print("-" * 60)
print(f"Total time          : {t4-t0:.4f} s")
print("-" * 60)

print()
print("Best Candidate")
print(f"Period   : {candidate['period']:.6f}")
print(f"SDE      : {candidate['sde']:.3f}")
print(f"Depth    : {candidate['depth']:.6f}")
print(f"Duration : {candidate['duration']:.6f}")

print()
print("Feature count :", len(features))

print("=" * 60)