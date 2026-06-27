from time import perf_counter
import numpy as np

from src.loader import load_kepler_data
from src.preprocessing import preprocess
from src.bls_numba import search_period
from src.features import (
    _phase1,
    _phase2,
    _phase3,
    _phase4
)

time, flux, quality = load_kepler_data(
    "data/raw/K00752.01.npz"
)

proc = preprocess(time, flux, quality)

result = search_period(
    proc["time"],
    proc["flux"],
    np.linspace(0.5,30,5000),
    nbins=200,
    min_duration=0.01,
    max_duration=0.10
)

candidate = result["best"]

t0 = perf_counter()
_phase1(candidate, proc["time"], proc["flux"])
t1 = perf_counter()

_phase2(candidate)
t2 = perf_counter()

_phase3(candidate, proc["time"], proc["flux"])
t3 = perf_counter()

_phase4(candidate, proc["flux"])
t4 = perf_counter()

print("Phase1:", t1-t0)
print("Phase2:", t2-t1)
print("Phase3:", t3-t2)
print("Phase4:", t4-t3)