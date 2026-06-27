import os
import time as time_module
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from src.loader import load_kepler_data
from src.preprocessing import preprocess
from src.bls_numba import search_period, mask_transit
from src.features import compute_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

LABEL_MAP = {
    "CONFIRMED":      0,
    "CANDIDATE":      1,
    "FALSE POSITIVE": 2
}

LABEL_NAMES = {
    0: "PLANET",
    1: "CANDIDATE",
    2: "FALSE_POSITIVE"
}


class PipelineConfig:
    def __init__(
        self,
        nbins=200,
        min_duration=0.01,
        max_duration=0.10,
        preprocess_window=51,
        preprocess_poly=3,
        coarse_factor=5,
        top_k=10,
        sde_threshold=3.0,
        min_transit_points=2,
        duty_min=0.0005,
        duty_max=0.9,
        n_planets=3,
        period_grid=None
    ):
        self.nbins = nbins
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.preprocess_window = preprocess_window
        self.preprocess_poly = preprocess_poly
        self.coarse_factor = coarse_factor
        self.top_k = top_k
        self.sde_threshold = sde_threshold
        self.min_transit_points = min_transit_points
        self.duty_min = duty_min
        self.duty_max = duty_max
        self.n_planets = n_planets
        self.period_grid = period_grid

    def to_dict(self):
        d = self.__dict__.copy()
        d.pop('period_grid', None)
        return d


class StarData:
    def __init__(self, kepid, clean_time, clean_flux, periodogram, planets):
        self.kepid = kepid
        self.clean_time = clean_time
        self.clean_flux = clean_flux
        self.periodogram = periodogram
        self.planets = planets  # list of (candidate, features) tuples


class RecoveryStats:
    def __init__(self, kepid):
        self.kepid = kepid
        self.recovered_planets = []
        self.missed_planets = []

    def add_recovery(self, true_period, recovered_period, koi_name):
        error = abs(recovered_period - true_period) / true_period

        harmonic = None
        for h in [0.5, 2.0, 1.0 / 3, 3.0]:
            if abs(recovered_period - true_period * h) / true_period < 0.1:
                harmonic = f"{h:.2f}P" if h != 1.0 else "P"
                break

        self.recovered_planets.append({
            "true_period": true_period,
            "recovered_period": recovered_period,
            "error": error,
            "harmonic": harmonic,
            "koi": koi_name
        })

    def add_miss(self, true_period, koi_name):
        self.missed_planets.append({
            "true_period": true_period,
            "koi": koi_name
        })


class DatasetSummary:
    def __init__(self):
        self.total_stars = 0
        self.total_rows = 0
        self.failed_stars = 0
        self.label_counts = defaultdict(int)
        self.sde_values = []
        self.processing_times = []
        self.recovery_stats = []
        self.missing_values = defaultdict(int)
        self.failure_reasons = defaultdict(int)

    def add_row(self, row):
        self.total_rows += 1
        self.label_counts[row.get("label_name", "UNKNOWN")] += 1
        self.sde_values.append(row.get("bls_sde", 0))
        self.processing_times.append(row.get("processing_time", 0))

        for key, value in row.items():
            if pd.isna(value) or value is None:
                self.missing_values[key] += 1

    def add_recovery(self, stats):
        self.recovery_stats.append(stats)

    def add_failure(self, reason):
        self.failure_reasons[reason] += 1

    def print_summary(self):
        logger.info("\n" + "=" * 70)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total stars processed: {self.total_stars}")
        logger.info(f"Successful: {self.total_rows}")
        logger.info(f"Failed: {self.failed_stars}")
        logger.info(f"\nLabel distribution:")
        for label, count in sorted(self.label_counts.items()):
            logger.info(f"  {label}: {count}")

        if self.sde_values:
            logger.info(f"\nSDE statistics:")
            logger.info(f"  Mean: {np.mean(self.sde_values):.2f}")
            logger.info(f"  Median: {np.median(self.sde_values):.2f}")
            logger.info(f"  Std: {np.std(self.sde_values):.2f}")

        if self.processing_times:
            logger.info(f"\nProcessing time:")
            logger.info(f"  Mean: {np.mean(self.processing_times):.3f}s")
            logger.info(f"  Total: {np.sum(self.processing_times):.1f}s")

        total_catalog = sum(
            len(s.missed_planets) + len(s.recovered_planets)
            for s in self.recovery_stats
        )
        total_recovered = sum(
            len(s.recovered_planets) for s in self.recovery_stats
        )
        if total_catalog > 0:
            logger.info(f"\nRecovery statistics:")
            logger.info(f"  Total catalog planets: {total_catalog}")
            logger.info(f"  Recovered: {total_recovered} ({100 * total_recovered / total_catalog:.1f}%)")
            if total_recovered > 0:
                avg_error = np.mean([
                    p['error']
                    for s in self.recovery_stats
                    for p in s.recovered_planets
                ])
                logger.info(f"  Average recovery error: {avg_error * 100:.2f}%")

                harmonics = {}
                for s in self.recovery_stats:
                    for p in s.recovered_planets:
                        if p['harmonic']:
                            harmonics[p['harmonic']] = harmonics.get(p['harmonic'], 0) + 1

                if harmonics:
                    logger.info(f"  Harmonic detections:")
                    for harmonic, count in sorted(harmonics.items()):
                        logger.info(f"    {harmonic}: {count}")

        if self.failure_reasons:
            logger.info(f"\nFailure reasons:")
            for reason, count in sorted(
                self.failure_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger.info(f"  {reason}: {count}")

        for stats in self.recovery_stats[:10]:
            if stats.recovered_planets or stats.missed_planets:
                logger.info(f"\nKIC {stats.kepid}:")
                for p in stats.recovered_planets:
                    harmonic_str = f" ({p['harmonic']})" if p['harmonic'] else ""
                    logger.info(
                        f"  Recovered: {p['koi']}: {p['true_period']:.4f}d"
                        f" → {p['recovered_period']:.4f}d"
                        f" (error: {p['error'] * 100:.2f}%{harmonic_str})"
                    )
                for p in stats.missed_planets:
                    logger.info(f"  Missed: {p['koi']}: {p['true_period']:.4f}d")

        logger.info("=" * 70 + "\n")


def match_candidates_to_kois_hungarian(candidates, catalog_rows):
    if not candidates or not catalog_rows:
        return [(c, None) for c in candidates]

    n_cand = len(candidates)
    n_koi = len(catalog_rows)

    cost_matrix = np.zeros((n_cand, n_koi))

    for i, candidate in enumerate(candidates):
        for j, koi in enumerate(catalog_rows):
            true_period = koi.get("koi_period")
            if pd.isna(true_period):
                cost_matrix[i, j] = 1e9
            else:
                period_error = abs(candidate['period'] - true_period) / true_period
                cost_matrix[i, j] = period_error

    cand_idx, koi_idx = linear_sum_assignment(cost_matrix)

    matches = {}
    for c_i, k_i in zip(cand_idx, koi_idx):
        if cost_matrix[c_i, k_i] < 0.5:
            matches[c_i] = catalog_rows[k_i]

    result = []
    for i, candidate in enumerate(candidates):
        matched_koi = matches.get(i)
        result.append((candidate, matched_koi))

    return result


def load_star(filepath):
    try:
        time, flux, quality = load_kepler_data(filepath)
        return time, flux, quality
    except Exception as e:
        logger.debug(f"Failed to load {filepath}: {e}")
        return None, None, None


def preprocess_star(time, flux, quality, config):
    try:
        proc = preprocess(
            time, flux, quality,
            window=config.preprocess_window,
            poly=config.preprocess_poly
        )
        return proc["time"], proc["flux"]
    except Exception as e:
        logger.debug(f"Preprocessing failed: {e}")
        return None, None


def run_bls(clean_time, clean_flux, config):
    try:
        result = search_period(
            time=clean_time,
            flux=clean_flux,
            period_grid=config.period_grid,
            nbins=config.nbins,
            min_duration=config.min_duration,
            max_duration=config.max_duration,
            coarse_factor=config.coarse_factor,
            top_k=config.top_k,
            sde_threshold=config.sde_threshold,
            min_transit_points=config.min_transit_points,
            duty_min=config.duty_min,
            duty_max=config.duty_max
        )
        return result
    except Exception as e:
        logger.debug(f"BLS failed: {e}")
        return None


def detect_planets_iterative(clean_time, clean_flux, initial_result, config):
    """
    Multi-planet detection with iterative masking.
    Reuses initial BLS result for the first planet.
    Subsequent planets re-run BLS on masked flux.
    Rejects duplicates by period proximity.
    """
    planets = []
    working_flux = clean_flux.copy()

    if initial_result and initial_result.get("candidates"):
        candidate = initial_result["candidates"][0]
        features = compute_features(candidate, clean_time, working_flux)
        planets.append((candidate, features))

        transit_mask = mask_transit(
            clean_time,
            candidate['period'],
            candidate['epoch'],
            candidate['duration']
        )
        noise = np.std(working_flux[~transit_mask])
        baseline = np.median(working_flux[~transit_mask])
        working_flux[transit_mask] = (
            baseline + np.random.normal(0, noise, np.sum(transit_mask))
        )

    for planet_num in range(1, config.n_planets):
        # FIX: pass clean_time alongside working_flux
        result = run_bls(clean_time, working_flux, config)

        if result is None or not result.get("candidates"):
            break

        accepted = None

        for candidate in result["candidates"]:
            is_duplicate = False
            for prev_candidate, _ in planets:
                period_error = (
                    abs(candidate['period'] - prev_candidate['period'])
                    / prev_candidate['period']
                )
                if period_error < 0.01:
                    logger.debug(
                        f"Skipping duplicate period {candidate['period']:.4f}d"
                        f" (matches {prev_candidate['period']:.4f}d)"
                    )
                    is_duplicate = True
                    break
            if not is_duplicate:
                accepted = candidate
                break

        if accepted is None:
            break

        features = compute_features(accepted, clean_time, working_flux)
        planets.append((accepted, features))

        transit_mask = mask_transit(
            clean_time,
            accepted['period'],
            accepted['epoch'],
            accepted['duration']
        )
        noise = np.std(working_flux[~transit_mask])
        baseline = np.median(working_flux[~transit_mask])
        working_flux[transit_mask] = (
            baseline + np.random.normal(0, noise, np.sum(transit_mask))
        )

    return planets


def process_star_complete(kepid, filepath, catalog_rows, config, cache_dir="data/cache"):
    t_start = time_module.time()

    cache_file = os.path.join(cache_dir, f"KIC{kepid}.pkl") if cache_dir else None

    if cache_file and os.path.exists(cache_file):
        try:
            import pickle
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Loaded cache for KIC {kepid}")
            return data, time_module.time() - t_start, None
        except Exception as e:
            logger.debug(f"Cache load failed for KIC {kepid}: {e}")

    time, flux, quality = load_star(filepath)
    if time is None:
        return None, time_module.time() - t_start, "LOAD_FAILED"

    clean_time, clean_flux = preprocess_star(time, flux, quality, config)
    if clean_time is None or len(clean_time) < 100:
        return None, time_module.time() - t_start, "INSUFFICIENT_DATA"

    result = run_bls(clean_time, clean_flux, config)
    if result is None or not result.get("candidates"):
        return None, time_module.time() - t_start, "BLS_FAILED"

    planets = detect_planets_iterative(clean_time, clean_flux, result, config)

    if not planets:
        return None, time_module.time() - t_start, "NO_PLANETS"

    star_data = StarData(
        kepid=kepid,
        clean_time=clean_time,
        clean_flux=clean_flux,
        periodogram=result["periodogram"],
        planets=planets
    )

    if cache_file:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            import pickle
            with open(cache_file, "wb") as f:
                pickle.dump(star_data, f)
        except Exception as e:
            logger.debug(f"Cache save failed for KIC {kepid}: {e}")

    return star_data, time_module.time() - t_start, None


def create_dataset_rows(star_data, catalog_rows, config, processing_time):
    rows = []
    recovery = RecoveryStats(star_data.kepid)

    candidates = [p[0] for p in star_data.planets]

    matches = match_candidates_to_kois_hungarian(candidates, catalog_rows)

    matched_kois = set()

    for candidate_rank, ((candidate, features), (_, cat_row)) in enumerate(
        zip(star_data.planets, matches), start=1
    ):
        if cat_row is None:
            cat_row = {}
        else:
            matched_kois.add(cat_row.get("kepoi_name"))

        disposition = str(cat_row.get("koi_disposition", "")).upper()
        label = LABEL_MAP.get(disposition, 1)

        row = {
            "kepid":           star_data.kepid,
            "kepoi_name":      cat_row.get("kepoi_name"),
            "kepler_name":     cat_row.get("kepler_name"),
            "label":           label,
            "label_name":      LABEL_NAMES.get(label, "UNKNOWN"),
            "candidate_rank":  candidate_rank,
            "n_points":        len(star_data.clean_time),
            "processing_time": processing_time,

            "bls_period":   candidate['period'],
            "bls_epoch":    candidate['epoch'],
            "bls_duration": candidate['duration'],
            "bls_depth":    candidate['depth'],
            "bls_sr":       candidate['sr'],
            "bls_sde":      candidate['sde'],

            "true_period":   cat_row.get("koi_period"),
            "true_duration": cat_row.get("koi_duration"),
            "true_depth":    cat_row.get("koi_depth"),
            "true_radius":   cat_row.get("koi_prad"),

            "stellar_radius": cat_row.get("kstar_rad"),
            "stellar_temp":   cat_row.get("kstar_teff"),
            "stellar_logg":   cat_row.get("kstar_logg"),
            "kepmag":         cat_row.get("kepmag"),
            "koi_score":      cat_row.get("koi_score"),
            "koi_model_snr":  cat_row.get("koi_model_snr"),
            "koi_fpflag_nt":  cat_row.get("koi_fpflag_nt"),
            "koi_fpflag_ss":  cat_row.get("koi_fpflag_ss"),
            "koi_fpflag_co":  cat_row.get("koi_fpflag_co"),
            "koi_fpflag_ec":  cat_row.get("koi_fpflag_ec"),
            "ra":             cat_row.get("ra"),
            "dec":            cat_row.get("dec"),
        }

        row.update(features)
        rows.append(row)

        if cat_row:
            true_period = cat_row.get("koi_period")
            if not pd.isna(true_period):
                recovery.add_recovery(
                    true_period,
                    candidate['period'],
                    cat_row.get("kepoi_name")
                )

    for cat_row in catalog_rows:
        koi_name = cat_row.get("kepoi_name")
        if koi_name not in matched_kois:
            true_period = cat_row.get("koi_period")
            if not pd.isna(true_period):
                recovery.add_miss(true_period, koi_name)

    return rows, recovery


def _process_star_worker(args):
    kepid, filepath, catalog_rows, config, cache_dir = args

    star_data, proc_time, failure_reason = process_star_complete(
        kepid, filepath, catalog_rows, config, cache_dir
    )

    if star_data is None:
        return [], None, failure_reason, proc_time

    rows, recovery = create_dataset_rows(star_data, catalog_rows, config, proc_time)
    return rows, recovery, None, proc_time


def build_dataset(
    catalog_df,
    period_grid,
    data_dir="data/raw",
    config=None,
    kepid_col="kepid",
    cache_dir="data/cache",
    n_workers=None
):
    if config is None:
        config = PipelineConfig(period_grid=period_grid)
    else:
        config.period_grid = period_grid

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 8)

    os.makedirs(cache_dir, exist_ok=True)

    grouped = catalog_df.groupby(kepid_col)

    tasks = []
    for kepid, group_df in grouped:
        filepath = os.path.join(data_dir, f"KIC_{int(kepid)}.npz")

        if not os.path.exists(filepath):
            continue

        cat_rows = [row.to_dict() for _, row in group_df.iterrows()]

        tasks.append((
            int(kepid),
            filepath,
            cat_rows,
            config,
            cache_dir
        ))

    if not tasks:
        logger.warning("No matching data files found.")
        return pd.DataFrame(), DatasetSummary()

    logger.info(f"Processing {len(tasks)} stars with {n_workers} workers...")

    all_rows = []
    summary = DatasetSummary()
    summary.total_stars = len(tasks)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_star_worker, task) for task in tasks]

        for future in tqdm(futures, total=len(futures), desc="Processing", unit="star"):
            rows, recovery, failure_reason, proc_time = future.result()

            if failure_reason:
                summary.add_failure(failure_reason)
                summary.failed_stars += 1
            else:
                for row in rows:
                    summary.add_row(row)
                all_rows.extend(rows)
                if recovery:
                    summary.add_recovery(recovery)

    if not all_rows:
        logger.warning("No successful processing.")
        summary.print_summary()
        return pd.DataFrame(), summary

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["label", "kepid"]).reset_index(drop=True)

    summary.print_summary()

    return df, summary


def save_dataset(df, output_dir="datasets", base_name="dataset"):
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to save.")
        return

    os.makedirs(output_dir, exist_ok=True)

    csv_path     = os.path.join(output_dir, f"{base_name}.csv")
    parquet_path = os.path.join(output_dir, f"{base_name}.parquet")
    pkl_path     = os.path.join(output_dir, f"{base_name}.pkl")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)

    logger.info(f"Saved {len(df)} rows:")
    logger.info(f"  - {csv_path}")
    logger.info(f"  - {parquet_path}")
    logger.info(f"  - {pkl_path}")

    try:
        from sklearn.preprocessing import StandardScaler

        exclude_cols = {
            "kepid", "kepoi_name", "kepler_name", "label", "label_name",
            "true_period", "true_duration", "true_depth", "true_radius",
            "ra", "dec", "candidate_rank", "n_points", "processing_time"
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        df_scaled = df.copy()
        scaler = StandardScaler()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

        csv_scaled     = os.path.join(output_dir, f"{base_name}_scaled.csv")
        parquet_scaled = os.path.join(output_dir, f"{base_name}_scaled.parquet")

        df_scaled.to_csv(csv_scaled, index=False)
        df_scaled.to_parquet(parquet_scaled, index=False)

        logger.info(f"  - {csv_scaled}")
        logger.info(f"  - {parquet_scaled}")
    except Exception as e:
        logger.warning(f"Could not save scaled version: {e}")


def main(
    catalog_df,
    period_grid,
    data_dir="data/raw",
    output_dir="datasets",
    config=None,
    n_workers=None
):
    logger.info("Starting dataset building pipeline...")
    logger.info(f"Loaded catalog: {len(catalog_df)} rows")

    if config is None:
        config = PipelineConfig(period_grid=period_grid)
    else:
        config.period_grid = period_grid

    logger.info(f"Configuration: {config.to_dict()}")

    df, summary = build_dataset(
        catalog_df,
        period_grid,
        data_dir=data_dir,
        config=config,
        n_workers=n_workers
    )

    if df.empty:
        logger.error("Failed to build dataset.")
        return None, summary

    save_dataset(df, output_dir=output_dir)

    logger.info("Dataset building complete!")

    return df, summary


if __name__ == "__main__":
    catalog_df = pd.read_csv("data/kois.csv")

    period_grid = np.linspace(0.5, 30.0, 2000)

    config = PipelineConfig(
        nbins=200,
        n_planets=3,
        sde_threshold=3.0
    )

    df, summary = main(
        catalog_df,
        period_grid,
        data_dir="data/raw",
        output_dir="datasets",
        config=config,
        n_workers=8
    )