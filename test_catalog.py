from src.catalog import load_catalog

df = load_catalog()

print(
    df[
        ["kepid","kepoi_name","kepler_name"]
    ].head(10)
)