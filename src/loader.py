import numpy as np
def load_kepler_data(filename):
    data = np.load(filename)
    time = data["time"]
    flux = data["flux"]
    quality = data["quality"]

    return time, flux, quality

