import os
import time
import pandas as pd

from src.loader import load_kepler_data
from src.preprocessing import preprocess
from src.bls_numba import search_period
from src.plotting import save_all_plots


def process_target(
    filepath,
    period_grid,
    nbins,
    min_duration,
    max_duration,
    output_dir="outputs"
):
    """
    Process a single light curve.
    """

    target = os.path.splitext(
        os.path.basename(filepath)
    )[0]

    time_data, flux, quality = load_kepler_data(
        filepath
    )

    pre = preprocess(
        time_data,
        flux,
        quality
    )
    time_data = pre["time"]
    flux = pre["flux"]

    flux = flux - flux.mean()

    start = time.perf_counter()

    result = search_period(
        time_data,
        flux,
        period_grid,
        nbins,
        min_duration,
        max_duration
    )

    runtime = time.perf_counter() - start

    save_all_plots(
        result,
        output_dir,
        target
    )

    summary = {
        "Target": target,
        "Period": result["best_period"],
        "Epoch": result["best_epoch"],
        "Duration": result["best_duration"],
        "Depth": result["best_depth"],
        "SR": result["best_sr"],
        "SDE": result["sde"],
        "Runtime": runtime
    }

    return summary


def process_directory(
    input_dir,
    period_grid,
    nbins,
    min_duration,
    max_duration,
    output_dir="outputs"
):
    """
    Process every NPZ file in a directory.
    """

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".npz")
    )

    summaries = []

    for file in files:

        print(f"Processing {file}")

        summary = process_target(
            os.path.join(input_dir, file),
            period_grid,
            nbins,
            min_duration,
            max_duration,
            output_dir
        )

        summaries.append(summary)

    df = pd.DataFrame(summaries)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    df.to_csv(
        os.path.join(
            output_dir,
            "summary.csv"
        ),
        index=False
    )

    print(
        f"\nProcessed {len(df)} targets."
    )

    return df