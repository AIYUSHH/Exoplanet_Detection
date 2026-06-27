import numpy as np
def design_matrix(window_size, poly_degree):
    half = window_size//2
    x = np.arange(-half, half+1)
    A = np.zeros((window_size, poly_degree+1))
    for j in range(poly_degree+1):
        A[:, j] = x**j;
    return A


def savgol_coeff(window_size, poly_degree):
    A = design_matrix(window_size, poly_degree)
    ATA = A.T @ A
    ATA_inv = np.linalg.pinv(ATA) # Moore - Penrose pseudo_inverse (Numerically more stable)
    P = ATA_inv@A.T
    coefficients = P[0]
    return coefficients

def savgol_filter(signal, window_size, poly_degree):

    coeff = savgol_coeff(window_size, poly_degree)

    half = window_size // 2
    n = len(signal)

    filtered = np.empty(n)
    filtered[:] = np.nan

    for i in range(half, n-half):
        window = signal[i-half:i+half+1]
        filtered[i] = np.dot(window, coeff)

    return filtered