import lightkurve as lk
import numpy as np
import os
os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)
print("Searching Kepler Archive")
search = lk.search_lightcurve("Kepler-10", mission="Kepler")
print(search)
print("\nDownloading . . .\n")
lc = search[4].download()
print("Download Complete")
time = lc.time.value
flux = lc.flux.value
quality = np.array(lc.quality)
np.savez(
    "../data/raw/kepler10_raw.npz",
    time = time,
    flux = flux,
    quality = quality
)
print("\nSaved to data/raw/kepler10_raw.npz")