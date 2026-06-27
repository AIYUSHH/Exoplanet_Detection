import os
import matplotlib.pyplot as plt
import numpy as np


def plot_periodogram(result, save_path=None, show=False):
    """
    Plot the BLS periodogram.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(
        result["period_grid"],
        result["periodogram"],
        "k"
    )

    if result.get("best_period") is not None:
        plt.axvline(
            result["best_period"],
            color="red",
            linestyle="--",
            label="Best Period"
        )

    plt.xlabel("Period (days)")
    plt.ylabel("Signal Residue (SR)")
    plt.title("BLS Periodogram")
    plt.grid(True)
    if result.get("best_period") is not None:
        plt.legend()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close()


def plot_folded_curve(result, save_path=None, show=False):
    """
    Plot the folded light curve.
    """

    phase = result.get("best_phase")
    flux = result.get("best_flux")

    plt.figure(figsize=(10, 5))

    if phase is not None and flux is not None:
        plt.scatter(
            phase,
            flux,
            s=5,
            color="black"
        )

    epoch = result.get("best_epoch")
    duration = result.get("best_duration")

    if epoch is not None and duration is not None:
        plt.axvspan(
            epoch - duration / 2,
            epoch + duration / 2,
            color="red",
            alpha=0.3,
            label="Transit"
        )

    plt.xlabel("Phase")
    plt.ylabel("Relative Flux")
    plt.title("Folded Light Curve")
    plt.grid(True)
    if epoch is not None and duration is not None:
        plt.legend()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    plt.close()


def save_all_plots(
    result,
    output_dir,
    target_name
):
    """
    Save both diagnostic plots.
    """

    period_dir = os.path.join(
        output_dir,
        "periodograms"
    )

    folded_dir = os.path.join(
        output_dir,
        "folded_curves"
    )

    os.makedirs(period_dir, exist_ok=True)
    os.makedirs(folded_dir, exist_ok=True)

    plot_periodogram(
        result,
        os.path.join(
            period_dir,
            f"{target_name}.png"
        )
    )

    plot_folded_curve(
        result,
        os.path.join(
            folded_dir,
            f"{target_name}.png"
        )
    )