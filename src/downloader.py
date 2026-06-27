import os
import shutil
import numpy as np
import lightkurve as lk


def clear_lightkurve_cache():
    """
    Remove cached Lightkurve downloads.
    Useful if an interrupted download leaves corrupted FITS files.
    """
    cache_dir = os.path.expanduser("~/.cache/lightkurve")

    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("[CACHE] Cleared Lightkurve cache")
        except Exception as e:
            print(f"[CACHE] Failed to clear cache: {e}")


def download_target(
    kepid,
    mission="Kepler",
    cadence="long",
    output_dir="data/raw",
    overwrite=False,
    max_retries=2
):
    """
    Download ONE light curve for a single KEPID.
    """

    os.makedirs(output_dir, exist_ok=True)

    search_target = f"KIC {int(kepid)}"

    filepath = os.path.join(
        output_dir,
        f"KIC_{int(kepid)}.npz"
    )

    if os.path.exists(filepath) and not overwrite:
        print(f"[SKIP] {search_target}")
        return filepath

    print(f"[DOWNLOAD] {search_target}")

    for attempt in range(max_retries):

        try:

            search = lk.search_lightcurve(
                search_target,
                mission=mission,
                cadence=cadence
            )

            if len(search) == 0:
                print("   No light curve found.")
                return None

            lc = search.download_all().stitch()

            np.savez(
                filepath,
                time=lc.time.value,
                flux=lc.flux.value,
                quality=lc.quality
            )

            print(f"   Saved -> {filepath}")

            return filepath

        except Exception as e:

            print(
                f"   FAILED ({attempt+1}/{max_retries}) : {e}"
            )

            if attempt < max_retries - 1:
                clear_lightkurve_cache()

    return None


def download_many(
    catalog_df,
    mission="Kepler",
    cadence="long",
    output_dir="data/raw",
    overwrite=False
):
    """
    Download one light curve per UNIQUE KEPID.
    """

    required = {"kepid"}

    missing = required - set(catalog_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    stars = (
        catalog_df[["kepid"]]
        .dropna()
        .drop_duplicates()
        .sort_values("kepid")
        .reset_index(drop=True)
    )

    downloaded = []

    total = len(stars)

    for i, row in stars.iterrows():

        print(f"\n[{i+1}/{total}]")

        path = download_target(
            kepid=row["kepid"],
            mission=mission,
            cadence=cadence,
            output_dir=output_dir,
            overwrite=overwrite
        )

        if path is not None:
            downloaded.append(path)

    print("\n--------------------------------")
    print(f"Unique Stars : {total}")
    print(f"Downloaded   : {len(downloaded)}")
    print(f"Failed       : {total-len(downloaded)}")
    print("--------------------------------")

    return downloaded