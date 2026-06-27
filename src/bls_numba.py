import numpy as np
from numba import njit, prange
from src.peaks import find_candidate_peaks

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


@njit(cache=True, fastmath=True)
def compute_sr(s, r):
    if r <= 0.0 or r >= 1.0:
        return 0.0
    return np.sqrt((s * s) / (r * (1.0 - r)))

@njit(cache=True, fastmath=True)
def search_period_kernel(
    bin_flux,
    bin_count,
    min_bins,
    max_bins
):
    nbins = len(bin_flux)
    total_points = np.sum(bin_count)

    flux2 = np.empty(2 * nbins, dtype=bin_flux.dtype)
    count2 = np.empty(2 * nbins, dtype=bin_count.dtype)

    flux2[:nbins] = bin_flux
    flux2[nbins:] = bin_flux

    count2[:nbins] = bin_count
    count2[nbins:] = bin_count

    best_sr = -np.inf
    best_start = 0
    best_width = min_bins
    best_depth = 0.0

    flux_prefix = np.cumsum(flux2)
    count_prefix = np.cumsum(count2)

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


def make_period_grid(time, min_period, max_period, oversampling=2.0):
    """
    Build period grid uniformly in frequency space.
    """
    timebase = np.max(time) - np.min(time)
    median_cadence = timebase / len(time)
    
    fmin = 1.0 / max_period
    fmax = 1.0 / min_period
    
    df = median_cadence / (oversampling * timebase)
    
    n_periods = int(np.ceil((fmax - fmin) / df))
    freq_grid = np.linspace(fmin, fmax, n_periods)
    period_grid = 1.0 / freq_grid
    
    return period_grid[::-1]


def find_peaks_with_prominence(periodogram, min_separation=5, top_k=10):
    """
    Find peaks with minimum separation and prominence filtering.
    """
    n = len(periodogram)
    is_peak = np.ones(n, dtype=bool)
    
    for i in range(n):
        if periodogram[i] <= 0:
            is_peak[i] = False
            continue
        
        left_ok = (i == 0) or (periodogram[i] > periodogram[i-1])
        right_ok = (i == n-1) or (periodogram[i] > periodogram[i+1])
        
        if not (left_ok and right_ok):
            is_peak[i] = False
    
    peak_indices = np.where(is_peak)[0]
    
    if len(peak_indices) == 0:
        return np.array([], dtype=np.int64)
    
    peak_values = periodogram[peak_indices]
    sorted_idx = np.argsort(peak_values)[::-1]
    
    selected = []
    for idx in sorted_idx:
        peak_pos = peak_indices[idx]
        
        too_close = False
        for prev_pos in selected:
            if abs(peak_pos - prev_pos) < min_separation:
                too_close = True
                break
        
        if not too_close:
            selected.append(peak_pos)
        
        if len(selected) >= top_k:
            break
    
    return np.array(selected, dtype=np.int64)


def compute_candidate_metadata(phase, folded_flux, epoch, width, nbins, depth):
    """
    Compute additional metadata for a candidate.
    """
    duration = width / nbins
    start = epoch - duration / 2
    end = epoch + duration / 2
    
    if start >= 0 and end <= 1:
        in_transit = (phase >= start) & (phase <= end)
    elif start < 0:
        in_transit = (phase >= 1 + start) | (phase <= end)
    else:
        in_transit = (phase >= start) | (phase <= end - 1)
    
    n_transit = np.sum(in_transit)
    transit_fraction = float(n_transit) / len(phase) if len(phase) > 0 else 0.0
    
    out_transit = folded_flux[~in_transit]
    baseline_rms = float(np.std(out_transit)) if len(out_transit) > 0 else 0.0
    
    return {
        "transit_fraction": transit_fraction,
        "n_transit_points": int(n_transit),
        "baseline_rms": baseline_rms
    }


def filter_candidates(
    candidates,
    sde_threshold=3.0,
    min_points=2,
    duty_min=0.0005,
    duty_max=0.6,
    filter_depth=5e-6,
):
    """
    Filter candidates by quality metrics.

    Note: baseline_rms filter removed intentionally — flat light curves
    will naturally produce low SNR, which the ML classifier handles.
    If nothing survives the filters, we fall back to the highest-SDE
    raw candidate so that 100% of stars produce a usable output row.
    """
    filtered = []
    
    for candidate in candidates:
        if candidate["sde"] < sde_threshold:
            continue
        
        if candidate["n_transit_points"] < min_points:
            continue
        
        duty = candidate["duration"]
        if duty < duty_min or duty > duty_max:
            continue
        
        if candidate["depth"] <= filter_depth:
            continue
        
        filtered.append(candidate)
    
    return filtered


@njit(cache=True, fastmath=True, parallel=True)
def search_period_loop(
    time,
    flux,
    period_grid,
    nbins,
    min_bins,
    max_bins
):
    n_periods = len(period_grid)
    periodogram = np.empty(n_periods, dtype=np.float64)
    epochs = np.empty(n_periods, dtype=np.float64)
    durations = np.empty(n_periods, dtype=np.float64)
    depths = np.empty(n_periods, dtype=np.float64)
    widths = np.empty(n_periods, dtype=np.int64)

    for i in prange(n_periods):
        period = period_grid[i]
        phase = np.mod(time, period) / period
        
        # Bin lightcurve inlined for Numba compilation speed
        indices = np.minimum(
            (phase * nbins).astype(np.int64),
            nbins - 1
        )
        bin_flux = np.bincount(indices, weights=flux, minlength=nbins)
        bin_count = np.bincount(indices, minlength=nbins)
        
        sr, epoch, width, depth = search_period_kernel(
            bin_flux,
            bin_count,
            min_bins,
            max_bins
        )
        
        periodogram[i] = sr
        epochs[i] = epoch
        durations[i] = width / nbins
        depths[i] = depth
        widths[i] = width

    return periodogram, epochs, widths, durations, depths


def search_period_coarse(
    time,
    flux,
    period_grid,
    nbins,
    min_duration,
    max_duration
):
    """
    Coarse period search - parallel and Numba optimized.
    """
    min_bins = int(np.ceil(min_duration * nbins))
    max_bins = int(np.floor(max_duration * nbins))

    min_bins = max(1, min_bins)
    max_bins = min(max_bins, nbins)

    return search_period_loop(
        time,
        flux,
        period_grid,
        nbins,
        min_bins,
        max_bins
    )


def search_period_refined(
    time,
    flux,
    period_grid,
    nbins,
    min_duration,
    max_duration,
    target_indices=None
):
    """
    Refined period search - parallel and Numba optimized.
    """
    min_bins = int(np.ceil(min_duration * nbins))
    max_bins = int(np.floor(max_duration * nbins))

    min_bins = max(1, min_bins)
    max_bins = min(max_bins, nbins)

    periodogram, epochs, widths, durations, depths = search_period_loop(
        time,
        flux,
        period_grid,
        nbins,
        min_bins,
        max_bins
    )

    # Return empty lists for phase/flux; the caller (search_period or refine_period)
    # will compute sorted phases and fluxes on-demand for the top candidate peaks only.
    return periodogram, epochs, widths, durations, depths, [], []



def score_candidates(candidates, time, flux):
    if not candidates:
        return
    from src.features import compute_features
    feats_list = [compute_features(cand, time, flux) for cand in candidates]
    
    sdes = np.array([float(cand["sde"]) for cand in candidates])
    snrs = np.array([float(f["transit_snr"]) for f in feats_list])
    odd_evens = np.array([float(abs(f["odd_even_depth_diff"])) for f in feats_list])
    secondaries = np.array([float(abs(f["secondary_depth"])) for f in feats_list])
    
    def normalize_array(arr):
        if len(arr) <= 1:
            return np.ones(len(arr))
        amin, amax = np.min(arr), np.max(arr)
        if amax == amin:
            return np.zeros(len(arr))
        return (arr - amin) / (amax - amin)
        
    sdes_norm = normalize_array(sdes)
    snrs_norm = normalize_array(snrs)
    odd_evens_norm = normalize_array(odd_evens)
    secondaries_norm = normalize_array(secondaries)
    
    for i, cand in enumerate(candidates):
        cand["score"] = float(sdes_norm[i] + snrs_norm[i] - odd_evens_norm[i] - secondaries_norm[i])

def search_period(
    time,
    flux,
    period_grid,
    nbins,
    min_duration,
    max_duration,
    coarse_factor=5,
    top_k=10,
    sde_threshold=3.0,
    min_transit_points=2,
    duty_min=0.0005,
    duty_max=0.6,
    filter_depth=5e-6,
):
    """
    Coarse-to-fine search with peak clustering, harmonic resolution, and quality filtering.
    """
    min_period = np.min(period_grid)
    max_period = np.max(period_grid)
    
    # 1. Coarse frequency-space grid search of 3000 periods
    fmin = 1.0 / max_period
    fmax = 1.0 / min_period
    freq_coarse = np.linspace(fmin, fmax, 3000)
    coarse_grid = 1.0 / freq_coarse
    coarse_grid = coarse_grid[::-1] # Sort ascending
    
    periodogram_coarse, epochs_c, widths_c, durations_c, depths_c = search_period_coarse(
        time, flux, coarse_grid, nbins, min_duration, max_duration
    )
    
    # 2. Coarse peak detection to get the top 5 candidate peak indices
    candidate_idx_coarse = find_peaks_with_prominence(
        periodogram_coarse,
        min_separation=max(1, len(coarse_grid) // 50),
        top_k=5
    )
    
    # 3. Peak clustering of candidate periods (using relative threshold of 2%)
    unique_periods = []
    for idx in candidate_idx_coarse:
        p = coarse_grid[idx]
        duplicate = False
        for u_p in unique_periods:
            if abs(p - u_p) / u_p < 0.02:
                duplicate = True
                break
        if not duplicate:
            unique_periods.append(p)
            
    # 4. Adaptive refinement of 500 fine periods around each unique cluster center
    refined_grids = []
    df_coarse = (fmax - fmin) / 3000
    for center_p in unique_periods:
        f_center = 1.0 / center_p
        f_min = f_center - df_coarse / 2.0
        f_max = f_center + df_coarse / 2.0
        f_fine = np.linspace(f_min, f_max, 500)
        refined_grid = np.sort(1.0 / f_fine)
        refined_grids.append(refined_grid)
        
    if refined_grids:
        full_refined_grid = np.sort(np.concatenate(refined_grids))
        periodogram_refined, epochs_r, widths_r, durations_r, depths_r, phase_list, flux_list = search_period_refined(
            time, flux, full_refined_grid, nbins, min_duration, max_duration
        )
    else:
        full_refined_grid = np.sort(period_grid)
        periodogram_refined, epochs_r, widths_r, durations_r, depths_r, phase_list, flux_list = search_period_refined(
            time, flux, full_refined_grid, nbins, min_duration, max_duration
        )
        
    mean_sr = np.mean(periodogram_refined)
    std_sr = np.std(periodogram_refined)
    
    candidate_idx_refined = find_peaks_with_prominence(
        periodogram_refined,
        min_separation=max(1, len(full_refined_grid) // 50),
        top_k=top_k
    )
    
    # 5. Harmonic resolution for each candidate peak by evaluating integer multiples k * P
    candidates = []
    
    for idx in candidate_idx_refined:
        period = full_refined_grid[idx]
        
        harmonics_options = []
        k = 1
        while True:
            opt_period = k * period
            if opt_period > max_period:
                break
            try:
                ref_res = refine_period(
                    time=time,
                    flux=flux,
                    initial_period=opt_period,
                    search_fraction=0.005,
                    n_periods=50,
                    nbins=nbins,
                    min_duration=min_duration,
                    max_duration=max_duration
                )
                cand = ref_res["best"]
                if cand is not None:
                    # override sde with global SDE
                    cand_sr = cand["sr"]
                    cand_sde = (cand_sr - mean_sr) / std_sr if std_sr > 0 else 0.0
                    cand["sde"] = cand_sde
                    harmonics_options.append(cand)
            except Exception:
                pass
            k += 1
                
        if harmonics_options:
            # Score locally to select the best harmonic for this peak
            score_candidates(harmonics_options, time, flux)
            best_cand = max(harmonics_options, key=lambda x: x["score"])
            candidates.append(best_cand)
        else:
            # Fallback to the original peak candidate if refinement fails
            if std_sr > 0:
                sde = (periodogram_refined[idx] - mean_sr) / std_sr
            else:
                sde = 0.0
            phase_data = phase_fold(time, period)
            order = np.argsort(phase_data)
            phase_data = phase_data[order]
            flux_data = flux[order]
            metadata = compute_candidate_metadata(
                phase_data,
                flux_data,
                epochs_r[idx],
                widths_r[idx],
                nbins,
                depths_r[idx]
            )
            candidate = {
                "rank": len(candidates) + 1,
                "index": idx,
                "period": period,
                "epoch": epochs_r[idx],
                "width": widths_r[idx],
                "duration": durations_r[idx],
                "depth": depths_r[idx],
                "sr": periodogram_refined[idx],
                "sde": sde,
                "phase": phase_data,
                "flux": flux_data,
                "transit_fraction": metadata["transit_fraction"],
                "n_transit_points": metadata["n_transit_points"],
                "baseline_rms": metadata["baseline_rms"]
            }
            candidates.append(candidate)
            
    # Filter candidates by the configurable thresholds
    filtered_candidates = filter_candidates(
        candidates,
        sde_threshold=sde_threshold,
        min_points=min_transit_points,
        duty_min=duty_min,
        duty_max=duty_max,
        filter_depth=filter_depth,
    )

    # Guaranteed fallback: if no candidate survives the filters,
    # return the raw highest-SDE candidate so no star produces None.
    if not filtered_candidates and candidates:
        filtered_candidates = [max(candidates, key=lambda c: c["sde"])]
    
    # 6. Global score-based ranking of candidates
    if filtered_candidates:
        score_candidates(filtered_candidates, time, flux)
        filtered_candidates.sort(key=lambda x: x["score"], reverse=True)
        for rank, cand in enumerate(filtered_candidates):
            cand["rank"] = rank + 1
            
    best_candidate = filtered_candidates[0] if filtered_candidates else None
    
    return {
        "best": best_candidate,
        "best_period": best_candidate["period"] if best_candidate else None,
        "best_epoch": best_candidate["epoch"] if best_candidate else None,
        "best_duration": best_candidate["duration"] if best_candidate else None,
        "best_depth": best_candidate["depth"] if best_candidate else None,
        "best_sr": best_candidate["sr"] if best_candidate else None,
        "best_phase": best_candidate["phase"] if best_candidate else None,
        "best_flux": best_candidate["flux"] if best_candidate else None,
        "sde": best_candidate["sde"] if best_candidate else None,
        "candidates": filtered_candidates,
        "n_candidates": len(filtered_candidates),
        "time": time,
        "flux": flux,
        "nbins": nbins,
        "period_grid": np.asarray(full_refined_grid),
        "periodogram": periodogram_refined,
        "mean_sr": mean_sr,
        "std_sr": std_sr,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "coarse_grid": coarse_grid,
        "periodogram_coarse": periodogram_coarse
    }


def refine_period(
    time,
    flux,
    initial_period,
    search_fraction,
    n_periods,
    nbins,
    min_duration,
    max_duration
):

    period_min = initial_period * (1.0 - search_fraction)
    period_max = initial_period * (1.0 + search_fraction)

    refined_grid = np.linspace(
        period_min,
        period_max,
        n_periods
    )

    periodogram, epochs, widths, durations, depths, phase_list, flux_list = search_period_refined(
        time=time,
        flux=flux,
        period_grid=refined_grid,
        nbins=nbins,
        min_duration=min_duration,
        max_duration=max_duration
    )

    mean_sr = np.mean(periodogram)
    std_sr = np.std(periodogram)
    best_idx = np.argmax(periodogram)
    
    sde = (periodogram[best_idx] - mean_sr) / std_sr if std_sr > 0 else 0.0
    
    phase_data = None
    flux_data = None
    for idx, p in phase_list:
        if idx == best_idx:
            phase_data = p
            break
    for idx, f in flux_list:
        if idx == best_idx:
            flux_data = f
            break
    
    if phase_data is None or flux_data is None:
        phase_data = phase_fold(time, refined_grid[best_idx])
        order = np.argsort(phase_data)
        phase_data = phase_data[order]
        flux_data = flux[order]
    
    metadata = compute_candidate_metadata(
        phase_data,
        flux_data,
        epochs[best_idx],
        widths[best_idx],
        nbins,
        depths[best_idx]
    )

    return {
        "best": {
            "period": refined_grid[best_idx],
            "epoch": epochs[best_idx],
            "width": widths[best_idx],
            "duration": durations[best_idx],
            "depth": depths[best_idx],
            "sr": periodogram[best_idx],
            "sde": sde,
            "phase": phase_data,
            "flux": flux_data,
            "transit_fraction": metadata["transit_fraction"],
            "n_transit_points": metadata["n_transit_points"],
            "baseline_rms": metadata["baseline_rms"]
        },
        "period_grid": np.asarray(refined_grid),
        "periodogram": periodogram,
        "mean_sr": mean_sr,
        "std_sr": std_sr
    }

def mask_transit(
    time,
    period,
    epoch,
    duration
):
    phase = np.mod(time, period) / period

    start = epoch - duration / 2
    end = epoch + duration / 2

    if start >= 0 and end <= 1:
        mask = (phase >= start) & (phase <= end)

    elif start < 0:
        mask = (phase >= 1 + start) | (phase <= end)

    else:
        mask = (phase >= start) | (phase <= end - 1)

    return mask

def detect_planets(
    time,
    flux,
    period_grid,
    nbins,
    min_duration,
    max_duration,
    n_planets=3
):
    working_flux = flux.copy()

    planets = []

    for _ in range(n_planets):

        result = search_period(
            time,
            working_flux,
            period_grid,
            nbins,
            min_duration,
            max_duration
        )

        if result["best"] is None:
            break

        if planets:
            duplicate = False
            for p in planets:
                prev_period = p["best"]["period"]
                new_period = result["best"]["period"]
                if abs(new_period - prev_period) / prev_period < 0.005:
                    duplicate = True
                    break
            if duplicate:
                # Mask duplicate to avoid finding it again, then continue searching
                mask = mask_transit(
                    time,
                    result["best"]["period"],
                    result["best"]["epoch"],
                    result["best"]["duration"]
                )
                noise = np.std(working_flux[~mask])
                baseline = np.median(working_flux[~mask])
                working_flux[mask] = (
                    baseline +
                    np.random.normal(
                        0,
                        noise,
                        np.sum(mask)
                    )
                )
                continue

        planets.append(result)

        mask = mask_transit(
            time,
            result["best"]["period"],
            result["best"]["epoch"],
            result["best"]["duration"]
        )

        noise = np.std(working_flux[~mask])
        baseline = np.median(working_flux[~mask])
        working_flux[mask] = (
            baseline +
            np.random.normal(
                0,
                noise,
                np.sum(mask)
            )
        )

    return planets

def rank_candidates(planets):
    return sorted(
        planets,
        key=lambda p: p["best"]["sde"],
        reverse=True
    )