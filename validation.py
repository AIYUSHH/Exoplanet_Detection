import time
import numpy as np
import pandas as pd

from src.catalog import load_catalog
from src.loader import load_kepler_data
from src.preprocessing import preprocess
from src.bls_numba import search_period
from src.features import compute_features


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

DATA_DIR = "data/raw"

N_STARS = 10

NBINS = 200
MIN_DURATION = 0.01
MAX_DURATION = 0.10

freq = np.linspace(1 / 30, 1 / 0.5, 5000)
PERIOD_GRID = (1 / freq)[::-1]


# ----------------------------------------------------

def classify_period(true_p, recovered):

    ratio = recovered / true_p

    harmonics = {
        "P": 1.0,
        "P/2": 0.5,
        "2P": 2.0,
        "P/3": 1/3,
        "3P": 3.0
    }

    best = None
    err = 1e9

    for name, value in harmonics.items():
        e = abs(ratio - value)
        if e < err:
            err = e
            best = name

    return best


# ----------------------------------------------------

catalog = load_catalog()

catalog = catalog[
    catalog["koi_disposition"] == "CONFIRMED"
].head(N_STARS)

results = []

print("=" * 80)
print("PIPELINE VALIDATION")
print("=" * 80)

total_start = time.perf_counter()

for i, (_, row) in enumerate(catalog.iterrows(), start=1):

    star = row["kepoi_name"]

    file = f"{DATA_DIR}/{star}.npz"

    print(f"\n[{i}/{len(catalog)}] {star}")

    try:

        t0 = time.perf_counter()

        raw_time, raw_flux, quality = load_kepler_data(file)

        proc = preprocess(
            raw_time,
            raw_flux,
            quality,
            window=51,
            poly=3
        )

        clean_time = proc["time"]
        clean_flux = proc["flux"]

        result = search_period(
            clean_time,
            clean_flux,
            PERIOD_GRID,
            NBINS,
            MIN_DURATION,
            MAX_DURATION
        )

        candidate = result["best"]

        if candidate is None:
            raise RuntimeError("No candidate found.")

        features = compute_features(
            candidate,
            clean_time,
            clean_flux
        )

        runtime = time.perf_counter() - t0

        true_period = row["koi_period"]

        recovered = candidate["period"]

        rel_error = abs(recovered - true_period) / true_period

        harmonic = classify_period(
            true_period,
            recovered
        )

        nan_count = np.isnan(
            list(features.values())
        ).sum()

        inf_count = np.isinf(
            list(features.values())
        ).sum()

        passed = (
            nan_count == 0
            and inf_count == 0
        )

        results.append({

            "star": star,

            "true_period": true_period,

            "recovered_period": recovered,

            "relative_error": rel_error,

            "harmonic": harmonic,

            "sde": candidate["sde"],

            "runtime": runtime,

            "feature_count": len(features),

            "nan": nan_count,

            "inf": inf_count,

            "pass": passed

        })

        print(
            f"PASS | "
            f"P={recovered:.5f} "
            f"SDE={candidate['sde']:.2f} "
            f"Err={100*rel_error:.3f}% "
            f"{harmonic}"
        )

    except Exception as e:

        print("FAILED :", e)

        results.append({

            "star": star,

            "pass": False

        })

# ----------------------------------------------------

df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()

print(df)

print()

good = df["pass"].sum()

print(f"Stars tested        : {len(df)}")
print(f"Successful          : {good}")
print(f"Failed              : {len(df)-good}")

if good > 0:

    ok = df[df["pass"]]

    print()

    print(f"Mean runtime        : {ok['runtime'].mean():.3f} s")
    print(f"Median runtime      : {ok['runtime'].median():.3f} s")

    print()

    print(f"Mean rel error      : {100*ok['relative_error'].mean():.3f}%")
    print(f"Median rel error    : {100*ok['relative_error'].median():.3f}%")

    print()

    print("Harmonic recovery")

    print(ok["harmonic"].value_counts())

    print()

    print(f"Average SDE         : {ok['sde'].mean():.2f}")

    print(f"Average features    : {ok['feature_count'].mean():.1f}")

total_time = time.perf_counter() - total_start

print()
print(f"Total validation time : {total_time:.2f} s")

df.to_csv("validation_results.csv", index=False)

print()
print("Saved validation_results.csv")