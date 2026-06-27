import os
import pandas as pd

NASA_CATALOG_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/"
    "cgi-bin/nstedAPI/nph-nstedAPI?"
    "table=cumulative&format=csv"
)


def download_catalog(
    save_path="data/catalogs/koi_catalog.csv"
):
    """
    Download the latest KOI catalog from the
    NASA Exoplanet Archive.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    df = pd.read_csv(NASA_CATALOG_URL)

    df.to_csv(
        save_path,
        index=False
    )

    print(f"Saved catalog to {save_path}")

    return df


def load_catalog(
    filepath="data/catalogs/koi_catalog.csv"
):
    """
    Load the KOI catalog.
    """

    return pd.read_csv(filepath)


def get_confirmed_targets(df):
    """
    Return confirmed Kepler planets.
    """

    confirmed = df[
        df["koi_disposition"] == "CONFIRMED"
    ]

    return confirmed


def get_false_positive_targets(df):
    """
    Return false positives.
    """

    fp = df[
        df["koi_disposition"] == "FALSE POSITIVE"
    ]

    return fp


def get_candidate_targets(df):
    """
    Return planet candidates.
    """

    cand = df[
        df["koi_disposition"] == "CANDIDATE"
    ]

    return cand


def save_target_list(
    dataframe,
    filepath,
    column="kepler_name"
):
    """
    Save one target per line.

    Rows with missing names are skipped.
    """

    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    )

    targets = (
        dataframe[column]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    with open(filepath, "w") as f:

        for target in targets:

            f.write(target + "\n")

    print(
        f"Saved {len(targets)} targets to {filepath}"
    )