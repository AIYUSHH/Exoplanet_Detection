import numpy as np
from src.filters import savgol_filter


def remove_bad_quality(time, flux, quality):
    mask = (quality == 0)
    return time[mask], flux[mask]


def remove_nans(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]


def normalize_flux(flux):
    median = np.median(flux)
    return flux / median


def detrend_flux(flux, window=51, poly=3):

    trend = savgol_filter(
        flux,
        window,
        poly
    )

    detrended = flux / trend - 1

    return detrended, trend


def preprocess(
    time,
    flux,
    quality,
    window=51,
    poly=3
):

    # ---------------- Raw ----------------

    raw_time = time.copy()
    raw_flux = flux.copy()

    # ---------------- Quality ----------------

    quality_time, quality_flux = remove_bad_quality(
        raw_time,
        raw_flux,
        quality
    )

    # ---------------- NaN ----------------

    nan_time, nan_flux = remove_nans(
        quality_time,
        quality_flux
    )

    # ---------------- Normalize ----------------

    normalized_flux = normalize_flux(
        nan_flux
    )

    # ---------------- Detrend ----------------

    detrended_flux, trend = detrend_flux(
        normalized_flux,
        window,
        poly
    )

    mask = np.isfinite(detrended_flux)

    final_time = nan_time[mask]
    final_flux = detrended_flux[mask]
    final_trend = trend[mask]
    normalized_flux = normalized_flux[mask]

    return {
        "raw_time": raw_time,
        "raw_flux": raw_flux,

        "quality_time": quality_time,
        "quality_flux": quality_flux,

        "nan_time": nan_time,
        "nan_flux": nan_flux,

        "normalized_flux": normalized_flux,

        "trend": final_trend,

        "time": final_time,
        "flux": final_flux
    }