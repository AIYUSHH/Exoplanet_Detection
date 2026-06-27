from src.catalog import load_catalog
from src.downloader import download_many


def main():

    catalog = load_catalog()

    print("Catalog size :", len(catalog))

    download_many(
        catalog_df=catalog,
        mission="Kepler",
        cadence="long",
        output_dir="data/raw",
        overwrite=False
    )


if __name__ == "__main__":
    main()