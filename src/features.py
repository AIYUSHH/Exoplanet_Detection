import numpy as np
from scipy.stats import skew, kurtosis


def phase_mask(phase, epoch, duration):
    start = epoch - duration / 2
    end = epoch + duration / 2

    if start >= 0 and end <= 1:
        return (phase >= start) & (phase <= end)
    elif start < 0:
        return (phase >= 1 + start) | (phase <= end)
    else:
        return (phase >= start) | (phase <= end - 1)


def compute_features(candidate, time, flux):

    features = {}

    features.update(_phase1(candidate, time, flux))
    features.update(_phase2(candidate))
    features.update(_phase3(candidate, time, flux))
    features.update(_phase4(candidate, flux))

    # Add new features
    phase = candidate["phase"]
    folded_flux = candidate["flux"]
    epoch = candidate["epoch"]
    duration = candidate["duration"]
    depth = candidate["depth"]

    in_mask = phase_mask(phase, epoch, duration)
    in_transit = folded_flux[in_mask]
    out_transit = folded_flux[~in_mask]

    median_in_transit_flux = float(np.median(in_transit)) if len(in_transit) > 0 else 0.0
    median_out_transit_flux = float(np.median(out_transit)) if len(out_transit) > 0 else 0.0

    n_transit = len(in_transit)
    baseline_rms = float(np.std(out_transit)) if len(out_transit) > 1 else 0.0
    depth_uncertainty = float(baseline_rms / np.sqrt(n_transit)) if n_transit > 0 else 0.0
    transit_significance = float(abs(depth) / depth_uncertainty) if depth_uncertainty > 0 else 0.0

    local_baseline_slope = float(np.polyfit(phase[~in_mask], folded_flux[~in_mask], 1)[0]) if len(out_transit) >= 2 else 0.0

    rms = float(np.std(flux))
    W = 51
    if len(flux) >= W:
        x = np.asarray(flux)
        cumsum = np.cumsum(np.insert(x, 0, 0))
        cumsum2 = np.cumsum(np.insert(x**2, 0, 0))
        sum_w = cumsum[W:] - cumsum[:-W]
        sum2_w = cumsum2[W:] - cumsum2[:-W]
        mean_w = sum_w / W
        mean2_w = sum2_w / W
        var_w = np.clip(mean2_w - mean_w**2, 0.0, None)
        rolling_rms = float(np.mean(np.sqrt(var_w)))
    else:
        rolling_rms = rms

    white_noise = float(np.std(np.diff(flux)) / np.sqrt(2.0)) if len(flux) > 1 else 0.0
    red_noise = float(np.sqrt(max(0.0, rms**2 - white_noise**2)))

    features.update({
        "median_in_transit_flux": median_in_transit_flux,
        "median_out_transit_flux": median_out_transit_flux,
        "depth_uncertainty": depth_uncertainty,
        "transit_significance": transit_significance,
        "local_baseline_slope": local_baseline_slope,
        "rms": rms,
        "rolling_rms": rolling_rms,
        "white_noise": white_noise,
        "red_noise": red_noise
    })

    return features


def _phase1(candidate, time, flux):

    period = candidate["period"]
    epoch = candidate["epoch"]
    width = candidate["width"]
    duration = candidate["duration"]
    depth = candidate["depth"]
    sr = candidate["sr"]
    sde = candidate["sde"]
    phase = candidate["phase"]
    folded_flux = candidate["flux"]

    duty_cycle = duration

    transit_area = abs(depth) * duration

    in_mask = phase_mask(phase, epoch, duration)
    in_transit = folded_flux[in_mask]
    out_transit = folded_flux[~in_mask]

    transit_points = len(in_transit)
    baseline_median = float(np.median(out_transit)) if len(out_transit) > 0 else 0.0
    baseline_rms = float(np.std(out_transit)) if len(out_transit) > 1 else 0.0
    transit_rms = float(np.std(in_transit)) if len(in_transit) > 1 else 0.0

    if baseline_rms > 0 and transit_points > 0:
        transit_snr = float(abs(depth) * np.sqrt(transit_points) / baseline_rms)
    else:
        transit_snr = 0.0

    timebase = float(np.ptp(time))
    n_transits = int(np.floor(timebase / period)) if period > 0 else 0

    return {
        "period": period,
        "epoch": epoch,
        "width": int(width),
        "duration": duration,
        "depth": depth,
        "sr": sr,
        "sde": sde,
        "duty_cycle": duty_cycle,
        "transit_area": transit_area,
        "transit_points": transit_points,
        "baseline_median": baseline_median,
        "baseline_rms": baseline_rms,
        "transit_rms": transit_rms,
        "transit_snr": transit_snr,
        "n_transits": n_transits
    }


def _phase2(candidate):

    phase = candidate["phase"]
    folded_flux = candidate["flux"]
    epoch = candidate["epoch"]
    duration = candidate["duration"]
    depth = candidate["depth"]

    in_mask = phase_mask(phase, epoch, duration)
    in_phase = phase[in_mask]
    in_flux = folded_flux[in_mask]

    if len(in_flux) < 4:
        return {
            "ingress_slope": 0.0,
            "egress_slope": 0.0,
            "symmetry": 0.0,
            "flat_bottom_metric": 0.0,
            "v_shape_metric": 0.0
        }

    mid = epoch
    left_mask = in_phase < mid
    right_mask = in_phase >= mid

    left_phase = in_phase[left_mask]
    left_flux = in_flux[left_mask]
    right_phase = in_phase[right_mask]
    right_flux = in_flux[right_mask]

    if len(left_phase) >= 2:
        ingress_slope = float(np.polyfit(left_phase, left_flux, 1)[0])
    else:
        ingress_slope = 0.0

    if len(right_phase) >= 2:
        egress_slope = float(np.polyfit(right_phase, right_flux, 1)[0])
    else:
        egress_slope = 0.0

    if len(left_flux) > 0 and len(right_flux) > 0 and abs(depth) > 0:
        symmetry = float(abs(np.mean(left_flux) - np.mean(right_flux))) / abs(depth)
    else:
        symmetry = 0.0

    transit_depth = float(np.min(in_flux))
    transit_mean = float(np.mean(in_flux))
    if transit_depth != 0:
        flat_bottom_metric = float(abs(transit_mean - transit_depth) / abs(transit_depth))
    else:
        flat_bottom_metric = 0.0

    inner_mask = phase_mask(phase, epoch, duration / 2)
    outer_mask = in_mask & ~inner_mask

    inner_flux = folded_flux[inner_mask]
    outer_flux = folded_flux[outer_mask]

    if len(inner_flux) > 0 and len(outer_flux) > 0:
        v_shape_metric = float(abs(np.mean(inner_flux) - np.mean(outer_flux)))
        if abs(depth) > 0:
            v_shape_metric /= abs(depth)
    else:
        v_shape_metric = 0.0

    return {
        "ingress_slope": ingress_slope,
        "egress_slope": egress_slope,
        "symmetry": symmetry,
        "flat_bottom_metric": flat_bottom_metric,
        "v_shape_metric": v_shape_metric
    }


def _phase3(candidate, time, flux):

    period = candidate["period"]
    epoch = candidate["epoch"]
    duration = candidate["duration"]

    phase_full = np.mod(time, period) / period

    cycle_index = np.floor((time - epoch * period) / period).astype(np.int64)

    in_transit = phase_mask(phase_full, epoch, duration)

    odd_mask = in_transit & (cycle_index % 2 != 0)
    even_mask = in_transit & (cycle_index % 2 == 0)

    odd_flux = flux[odd_mask]
    even_flux = flux[even_mask]

    odd_depth = float(-np.mean(odd_flux)) if len(odd_flux) > 0 else 0.0
    even_depth = float(-np.mean(even_flux)) if len(even_flux) > 0 else 0.0

    odd_even_depth_diff = float(abs(odd_depth - even_depth))

    odd_epoch = float(np.mean(phase_full[odd_mask])) if len(odd_flux) > 0 else epoch
    even_epoch = float(np.mean(phase_full[even_mask])) if len(even_flux) > 0 else epoch

    odd_even_timing_diff = float(abs(odd_epoch - even_epoch))

    secondary_epoch = (epoch + 0.5) % 1.0
    sec_mask = phase_mask(phase_full, secondary_epoch, duration)
    secondary_flux = flux[sec_mask]
    secondary_depth = float(-np.mean(secondary_flux)) if len(secondary_flux) > 0 else 0.0

    half_period = period / 2.0
    phase_half = np.mod(time, half_period) / half_period
    half_epoch = (2 * epoch) % 1.0
    half_in_mask = phase_mask(phase_half, half_epoch, min(2 * duration, 1.0))
    half_in = flux[half_in_mask]
    half_out = flux[~half_in_mask]
    half_depth = float(-np.mean(half_in) + np.mean(half_out)) if len(half_in) > 0 and len(half_out) > 0 else 0.0

    double_period = period * 2.0
    phase_double = np.mod(time, double_period) / double_period
    double_epoch_a = epoch / 2.0
    double_epoch_b = (epoch + 1.0) / 2.0
    double_in_mask_a = phase_mask(phase_double, double_epoch_a, duration)
    double_in_mask_b = phase_mask(phase_double, double_epoch_b, duration)
    double_in_a = flux[double_in_mask_a]
    double_in_b = flux[double_in_mask_b]
    double_out_a = flux[~double_in_mask_a]
    double_out_b = flux[~double_in_mask_b]
    double_depth_a = float(-np.mean(double_in_a) + np.mean(double_out_a)) if len(double_in_a) > 0 and len(double_out_a) > 0 else 0.0
    double_depth_b = float(-np.mean(double_in_b) + np.mean(double_out_b)) if len(double_in_b) > 0 and len(double_out_b) > 0 else 0.0
    double_depth = max(double_depth_a, double_depth_b, key=abs)

    return {
        "odd_even_depth_diff": odd_even_depth_diff,
        "odd_even_timing_diff": odd_even_timing_diff,
        "secondary_depth": secondary_depth,
        "half_period_depth": half_depth,
        "double_period_depth": double_depth
    }


def _phase4(candidate, flux):

    folded_flux = candidate["flux"]
    
    folded_flux = folded_flux[np.isfinite(folded_flux)]
    
    if len(folded_flux) < 4:
        return {
            "mad": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "lag1_autocorrelation": 0.0,
            "local_variance": 0.0
        }

    mad = float(np.median(np.abs(folded_flux - np.median(folded_flux))))

    skewness = float(skew(folded_flux, bias=False))

    kurt = float(kurtosis(folded_flux, fisher=True, bias=False))

    if len(folded_flux) > 1:
        mean_flux = folded_flux.mean()
        centered = folded_flux - mean_flux
        den = np.dot(centered, centered)
        if den > 0:
            lag1_autocorrelation = float(np.dot(centered[:-1], centered[1:]) / den)
        else:
            lag1_autocorrelation = 0.0
    else:
        lag1_autocorrelation = 0.0

    window = max(1, len(folded_flux) // 10)
    trim = (len(folded_flux) // window) * window
    
    if trim > 0:
        blocks = folded_flux[:trim].reshape(-1, window)
        local_variance = float(np.var(blocks, axis=1).mean())
    else:
        local_variance = 0.0

    return {
        "mad": mad,
        "skewness": skewness,
        "kurtosis": kurt,
        "lag1_autocorrelation": lag1_autocorrelation,
        "local_variance": local_variance
    }

def extract_features(result):
    if result is None or result.get("best") is None:
        return {}
    
    features = compute_features(result["best"], result["time"], result["flux"])
    
    # Periodicity features from the periodogram
    if "periodogram" in result and "period_grid" in result:
        periodogram = result["periodogram"]
        period_grid = result["period_grid"]
        
        best_idx = np.argmax(periodogram)
        peak_val = periodogram[best_idx]
        mean_val = np.mean(periodogram)
        
        # 1. peak_prominence
        peak_prominence = float(peak_val - mean_val)
        
        # 2. peak_width (FWHM-like)
        HM = peak_val / 2.0
        left_idx = best_idx
        while left_idx > 0 and periodogram[left_idx] > HM:
            left_idx -= 1
        right_idx = best_idx
        while right_idx < len(periodogram) - 1 and periodogram[right_idx] > HM:
            right_idx += 1
        peak_width = float(period_grid[right_idx] - period_grid[left_idx])
        
        # 3. peak_isolation and neighbor_peak_ratio
        from src.bls_numba import find_peaks_with_prominence
        peaks_idx = find_peaks_with_prominence(
            periodogram,
            min_separation=max(1, len(periodogram) // 50),
            top_k=2
        )
        
        if len(peaks_idx) >= 2:
            best_peak_p = period_grid[peaks_idx[0]]
            second_peak_p = period_grid[peaks_idx[1]]
            peak_isolation = float(abs(best_peak_p - second_peak_p))
            neighbor_peak_ratio = float(periodogram[peaks_idx[1]] / periodogram[peaks_idx[0]])
        else:
            peak_isolation = 0.0
            neighbor_peak_ratio = 0.0
    else:
        peak_width = 0.0
        peak_prominence = 0.0
        peak_isolation = 0.0
        neighbor_peak_ratio = 0.0
        
    features.update({
        "peak_width": peak_width,
        "peak_prominence": peak_prominence,
        "peak_isolation": peak_isolation,
        "neighbor_peak_ratio": neighbor_peak_ratio
    })
    
    return features