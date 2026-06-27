import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

from src.loader import load_kepler_data
from src.preprocessing import preprocess
from src.bls_numba import search_period
from src.features import compute_features

# ============================================================
# CONFIGURATION
# ============================================================

FILE = "data/raw/K00752.01.npz"      # change to any downloaded target
CATALOG_PERIOD = 9.488               # known catalog period (optional)

# Generate period grid uniform in frequency space
fmin = 1.0 / 30.0
fmax = 1.0 / 0.5
freq_grid = np.linspace(fmin, fmax, 5000)
PERIOD_GRID = 1.0 / freq_grid
PERIOD_GRID = PERIOD_GRID[::-1]      # Sort ascending

NBINS = 200
MIN_DURATION = 0.01
MAX_DURATION = 0.10

if __name__ == '__main__':
    # ============================================================
    # START PROFILING
    # ============================================================

    tracemalloc.start()

    t0 = time.perf_counter()

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    t = time.perf_counter()
    time_raw, flux_raw, quality = load_kepler_data(FILE)
    load_time = time.perf_counter() - t

    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    t = time.perf_counter()

    proc = preprocess(
        time_raw,
        flux_raw,
        quality,
        window=51,
        poly=3
    )

    clean_time = proc["time"]
    clean_flux = proc["flux"]

    preprocess_time = time.perf_counter() - t

    # ------------------------------------------------------------
    # BLS
    # ------------------------------------------------------------

    t = time.perf_counter()

    result = search_period(
        time=clean_time,
        flux=clean_flux,
        period_grid=PERIOD_GRID,
        nbins=NBINS,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION
    )

    bls_time = time.perf_counter() - t

    candidate = result["best"]

    if candidate is None:
        raise RuntimeError("No candidate found.")

    # ------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------

    t = time.perf_counter()

    features = compute_features(
        candidate,
        clean_time,
        clean_flux
    )

    feature_time = time.perf_counter() - t

    total_time = time.perf_counter() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ============================================================
    # CHECKS
    # ============================================================

    assert len(candidate["phase"]) == len(candidate["flux"])
    assert candidate["period"] > 0
    assert candidate["duration"] > 0
    assert np.isfinite(list(features.values())).all()

    relative_error = abs(
        candidate["period"] - CATALOG_PERIOD
    ) / CATALOG_PERIOD

    # ============================================================
    # REPORT
    # ============================================================

    print("=" * 60)
    print("PIPELINE VALIDATION")
    print("=" * 60)

    print("\nDATA")
    print("-" * 60)
    print(f"Raw Samples      : {len(time_raw)}")
    print(f"Clean Samples    : {len(clean_time)}")
    print(f"Removed Samples  : {len(time_raw)-len(clean_time)}")

    print("\nRUNTIME")
    print("-" * 60)
    print(f"Load             : {load_time:.4f} s")
    print(f"Preprocess       : {preprocess_time:.4f} s")
    print(f"BLS              : {bls_time:.4f} s")
    print(f"Features         : {feature_time:.4f} s")
    print(f"Total            : {total_time:.4f} s")

    print("\nMEMORY")
    print("-" * 60)
    print(f"Current          : {current/1024/1024:.2f} MB")
    print(f"Peak             : {peak/1024/1024:.2f} MB")

    print("\nBEST CANDIDATE")
    print("-" * 60)
    print(f"Recovered Period : {candidate['period']:.6f} d")
    print(f"Catalog Period   : {CATALOG_PERIOD:.6f} d")
    print(f"Relative Error   : {100*relative_error:.3f}%")
    print(f"Epoch            : {candidate['epoch']:.5f}")
    print(f"Duration         : {candidate['duration']:.5f}")
    print(f"Depth            : {candidate['depth']:.8f}")
    print(f"SR               : {candidate['sr']:.3f}")
    print(f"SDE              : {candidate['sde']:.3f}")

    print("\nTOP CANDIDATES")
    print("-" * 60)

    for c in result["candidates"]:
        print(
            f"Rank {c['rank']:2d} | "
            f"P={c['period']:8.4f} | "
            f"SDE={c['sde']:7.2f} | "
            f"Depth={c['depth']:.6e}"
        )

    print("\nFEATURES")
    print("-" * 60)
    print(f"Feature Count : {len(features)}")
    print(f"NaNs          : {np.isnan(list(features.values())).sum()}")
    print(f"Infs          : {np.isinf(list(features.values())).sum()}")

    print("\nPIPELINE STATUS")
    print("-" * 60)
    print("Loader         : PASS")
    print("Preprocess     : PASS")
    print("BLS            : PASS")
    print("Features       : PASS")
    print("Overall        : PASS")

    print("=" * 60)

    # ============================================================
    # PLOTS
    # ============================================================

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # Raw Light Curve
    ax[0,0].scatter(time_raw, flux_raw, s=1)
    ax[0,0].set_title("Raw Light Curve")

    # Preprocessed
    ax[0,1].scatter(clean_time, clean_flux, s=1)
    ax[0,1].set_title("Preprocessed")

    # Periodogram
    ax[0,2].plot(
        result["period_grid"],
        result["periodogram"],
        lw=1
    )
    ax[0,2].axvline(
        candidate["period"],
        color="r",
        ls="--",
        label="Recovered"
    )
    ax[0,2].axvline(
        CATALOG_PERIOD,
        color="g",
        ls=":",
        label="Catalog"
    )
    ax[0,2].legend()
    ax[0,2].set_title("BLS Periodogram")

    # Phase Folded
    ax[1,0].scatter(
        candidate["phase"],
        candidate["flux"],
        s=2
    )
    ax[1,0].set_title("Phase Folded")

    # Transit Zoom
    mask = (
        np.abs(candidate["phase"] - candidate["epoch"])
        < 2 * candidate["duration"]
    )

    ax[1,1].scatter(
        candidate["phase"][mask],
        candidate["flux"][mask],
        s=4
    )
    ax[1,1].set_title("Transit Zoom")

    # Runtime
    labels = ["Load","Prep","BLS","Feat"]
    times = [
        load_time,
        preprocess_time,
        bls_time,
        feature_time
    ]

    ax[1,2].bar(labels, times)
    ax[1,2].set_ylabel("Seconds")
    ax[1,2].set_title("Runtime Breakdown")

    plt.tight_layout()
    # Save plot instead of showing it to avoid blocking execution in non-interactive environment
    plot_path = "plots/validation_result.png"
    import os
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved validation plot to {plot_path}")