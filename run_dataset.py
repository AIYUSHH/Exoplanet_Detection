import numpy as np

from src.batch import process_directory

period_grid = np.linspace(
    0.5,
    1.5,
    500
)

process_directory(
    input_dir="data/raw",
    period_grid=period_grid,
    nbins=200,
    min_duration=0.01,
    max_duration=0.08,
    output_dir="outputs"
)