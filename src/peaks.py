import numpy as np

def find_candidate_peaks(periodogram, top_k=10):

    peaks = []

    n = len(periodogram)

    for i in range(1, n-1):

        if (
            periodogram[i] > periodogram[i-1]
            and
            periodogram[i] >= periodogram[i+1]
        ):
            peaks.append(i)

    peaks.sort(
        key=lambda i: periodogram[i],
        reverse=True
    )

    return peaks[:top_k]