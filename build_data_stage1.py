import numpy as np
import pandas as pd
from src.catalog import load_catalog
from dataset_builder import build_star_dataset, save_dataset

if __name__ == '__main__':
    df_catalog = load_catalog()
    confirmed = df_catalog[df_catalog["koi_disposition"] == "CONFIRMED"].head(20)
    fp = df_catalog[df_catalog["koi_disposition"] == "FALSE POSITIVE"].head(20)
    small = pd.concat([confirmed, fp]).reset_index(drop=True)

    # Generate period grid uniform in frequency space
    fmin = 1.0 / 30.0
    fmax = 1.0 / 0.5
    freq_grid = np.linspace(fmin, fmax, 5000)
    period_grid = 1.0 / freq_grid
    period_grid = period_grid[::-1] # Sort ascending

    df = build_star_dataset(
        catalog_df=small,
        data_dir="data/raw",
        period_grid=period_grid
    )

    save_dataset(df, "datasets/dataset_small.csv")

    if not df.empty:
        print(df.head())
        print(df.info())
        print(df.describe())
        print(df["label_name"].value_counts())
        print(df.isna().sum().sort_values(ascending=False).head(20))