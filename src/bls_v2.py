import numpy as np


def phase_fold(time, period):
    phase = np.mod(time, period) / period
    return phase


def bin_lightcurve(phase, flux, nbins):
    indices = np.minimum(
        (phase * nbins).astype(np.int64),
        nbins - 1
    )
    bin_flux = np.bincount(
        indices,
        weights=flux,
        minlength=nbins
    )
    bin_count = np.bincount(
        indices,
        minlength=nbins
    )
    return bin_flux, bin_count


def compute_sr(s, r):
    if r <= 0.0 or r >= 1.0:
        return 0.0
    return np.sqrt((s * s) / (r * (1.0 - r)))


def search_period_kernel(
    bin_flux,
    bin_count,
    min_bins,
    max_bins
):
    nbins = len(bin_flux)
    total_points = np.sum(bin_count)

    flux2 = np.concatenate([bin_flux, bin_flux])
    count2 = np.concatenate([bin_count, bin_count])

    best_sr = -np.inf
    best_start = 0
    best_width = min_bins
    best_depth = 0.0

    flux_prefix = np.cumsum(flux2)
    count_prefix = np.cumsum(count2)

    # Precompute reciprocal to replace per-iteration division with multiplication
    inv_total = 1.0 / total_points

    for width in range(min_bins, max_bins + 1):
        current_flux = flux_prefix[width - 1]
        current_count = count_prefix[width - 1]
        r = float(current_count) * inv_total
        sr = compute_sr(current_flux, r)
        if current_count > 0:
            depth = -current_flux / current_count
        else:
            depth = 0.0
        if sr > best_sr:
            best_sr = sr
            best_start = 0
            best_width = width
            best_depth = depth

        for start in range(1, nbins):
            current_flux -= flux2[start - 1]
            current_flux += flux2[start + width - 1]
            current_count -= count2[start - 1]
            current_count += count2[start + width - 1]
            if current_count == 0:
                continue
            r = current_count * inv_total
            sr = compute_sr(current_flux, r)
            depth = -current_flux / current_count
            if sr > best_sr:
                best_sr = sr
                best_start = start
                best_width = width
                best_depth = depth

    best_epoch = ((best_start + best_width / 2) / nbins) % 1.0
    return (
        best_sr,
        best_epoch,
        best_width,
        best_depth
    )


def search_period(
    time,
    flux,
    period_grid,
    nbins,
    min_duration,
    max_duration
):
    # Initialize as numerics (not None) for Numba compatibility
    best_period = 0.0
    best_epoch = 0.0
    best_duration = 0.0
    best_depth = 0.0
    best_phase = np.empty(0)
    best_flux = np.empty(0)
    best_sr = -np.inf

    min_bins = int(np.ceil(min_duration * nbins))
    max_bins = int(np.floor(max_duration * nbins))

    min_bins = max(1, min_bins)
    max_bins = min(max_bins, nbins)

    periodogram = np.empty(len(period_grid), dtype=np.float64)

    for i, period in enumerate(period_grid):
        phase = phase_fold(time, period)

        # Inline sort (replaces sort_by_phase() call)
        order = np.argsort(phase)
        phase = phase[order]
        folded_flux = flux[order]

        bin_flux, bin_count = bin_lightcurve(phase, folded_flux, nbins)

        sr, epoch, width, depth = search_period_kernel(
            bin_flux,
            bin_count,
            min_bins,
            max_bins
        )

        duration = width / nbins
        periodogram[i] = sr

        if sr > best_sr:
            best_sr = sr
            best_period = period
            best_phase = phase
            best_flux = folded_flux
            best_epoch = epoch
            best_duration = duration
            best_depth = depth

    return {
        "best_period": best_period,
        "best_phase": best_phase,
        "best_flux": best_flux,
        "best_epoch": best_epoch,
        "best_duration": best_duration,
        "best_depth": best_depth,
        "best_sr": best_sr,
        "period_grid": np.asarray(period_grid),
        "periodogram": periodogram,
        "nbins": nbins
    }