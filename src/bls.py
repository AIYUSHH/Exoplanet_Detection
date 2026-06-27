import numpy as np
from src.prefix import prefix_sum, range_sum, interval_indices
def phase_fold(time, period):
    phase = np.mod(time,period)/period
    return phase

def sort_by_phase(phase, flux):
    order = np.argsort(phase)
    phase = phase[order]
    flux = flux[order]
    return phase, flux



def box_model(phase, epoch, duration):

    start = epoch - duration/2
    end = epoch + duration/2

    if start >= 0 and end <= 1:

        inside = (phase >= start) & (phase <= end)

    elif start < 0:

        inside = (phase >= 1 + start) | (phase <= end)

    else:

        inside = (phase >= start) | (phase <= end - 1)

    return inside

def box_score(
    prefix_flux,
    prefix_flux2,
    intervals,
    baseline
):

    S = 0.0
    Q = 0.0
    k = 0

    for left, right in intervals:

        if left > right:
            continue

        S += range_sum(
            prefix_flux,
            left,
            right
        )

        Q += range_sum(
            prefix_flux2,
            left,
            right
        )

        k += right - left + 1

    if k == 0:
        return None, np.inf, -np.inf

    depth = -S / k

    sse_inside = Q - (S*S)/k

    sse_outside = baseline - Q

    sse = sse_inside + sse_outside

    score = (baseline - sse) / baseline

    return depth, sse, score

from src.prefix import interval_indices

def search_epoch_duration(
    phase,
    prefix_flux,
    prefix_flux2,
    epoch_grid,
    duration_grid,
    baseline
):

    best_score = -np.inf
    best_sse = np.inf

    best_epoch = None
    best_duration = None
    best_depth = None
    best_intervals = None

    for epoch in epoch_grid:

        for duration in duration_grid:

            start = epoch - duration/2
            end = epoch + duration/2

            intervals = interval_indices(
                phase,
                start,
                end
            )

            depth, sse, score = box_score(
                prefix_flux,
                prefix_flux2,
                intervals,
                baseline
            )

            if score > best_score:

                best_score = score
                best_sse = sse

                best_epoch = epoch
                best_duration = duration
                best_depth = depth
                best_intervals = intervals

    return {
    "epoch": best_epoch,
    "duration": best_duration,
    "depth": best_depth,
    "score": best_score,
    "sse": best_sse,
    "intervals": best_intervals
    }

def search_period(
    time,
    flux,
    period_grid,
    epoch_grid_size,
    duration_grid,
    baseline
):

    best_score = -np.inf

    best_period = None
    best_phase = None
    best_flux = None
    best_result = None

    all_periods = []
    all_scores = []

    for period in period_grid:



        phase = phase_fold(
            time,
            period
        )

        phase, folded_flux = sort_by_phase(
            phase,
            flux
        )



        prefix_flux = prefix_sum(
            folded_flux
        )

        prefix_flux2 = prefix_sum(
            folded_flux**2
        )



        epoch_grid = np.linspace(
            0,
            1,
            epoch_grid_size
        )



        result = search_epoch_duration(
            phase,
            prefix_flux,
            prefix_flux2,
            epoch_grid,
            duration_grid,
            baseline
        )

        all_periods.append(period)
        all_scores.append(result["score"])


        if result["score"] > best_score:

            best_score = result["score"]

            best_period = period

            best_phase = phase

            best_flux = folded_flux

            best_result = result



    return {

        "period": best_period,

        "phase": best_phase,

        "flux": best_flux,

        "periods": all_periods,

        "scores": all_scores,

        "epoch": best_result["epoch"],

        "duration": best_result["duration"],

        "depth": best_result["depth"],

        "score": best_result["score"],

        "sse": best_result["sse"],

        "intervals": best_result["intervals"]

    }