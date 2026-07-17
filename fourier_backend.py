import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d

#default lambda for Cu source
LAMBDA_CU_KA = 0.15406

#default test file temporary
DEFAULT_FILE = r'C:\Users\dubec\OneDrive - UPJŠ\ERASMUS+\Data\XRR_Data\Simulated_TiO2_500.dat'

def two_theta_to_q(two_theta, wavelength=LAMBDA_CU_KA):
    theta = np.radians(two_theta / 2.0)
    return 4 * np.pi * np.sin(theta) / wavelength

def read_two_columns(file_path):
    """
    Reads data from a file. A bit overkill to make sure that any type of data file can be read,
    regardless of header type and size
    """
    x_vals, R_vals = [], []
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                x_vals.append(float(tokens[0]))
                R_vals.append(float(tokens[1]))
            except ValueError:
                continue
    x, R = np.array(x_vals), np.array(R_vals)
    valid = np.isfinite(x) & np.isfinite(R) & (R > 0)
    return x[valid], R[valid]

def load_reflectivity(file_path, wavelength=LAMBDA_CU_KA, axis='two_theta'):
    """
    Loading data with an optional transformation from degrees to A^-1
    """
    x, R = read_two_columns(file_path)
    if axis == 'q_angstrom':
        q = x * 10.0  # Angstrom^-1 -> nm^-1
    elif axis == 'two_theta':
        q = two_theta_to_q(x, wavelength)
    else:
        raise ValueError(f"Unknown axis '{axis}'; expected 'two_theta' or 'q_angstrom'.")
    return q, R

def find_critical_edge(q, R):
    """
    Finding a critical edge as steepest decend with an amplitude of first derivative done on a log10 scale
    """
    log_R = np.log10(R)
    dlogR_dq = np.gradient(log_R, q)
    idx = int(np.argmin(dlogR_dq))
    return idx, dlogR_dq

def find_noise_floor_cutoff(q, R, tail_fraction=0.05, n_sigma=5, min_run=3):
    n_tail = max(10, int(len(R) * tail_fraction))
    tail = R[-n_tail:]

    med = np.median(tail)
    mad = np.median(np.abs(tail - med))
    sigma = 1.4826 * mad
    threshold = med + n_sigma * sigma

    above = (R > threshold).astype(int)

    # window_sum[j] = number of True values in above[j : j+min_run]
    window_sum = np.convolve(above, np.ones(min_run, dtype=int), mode='valid')
    valid_starts = np.where(window_sum == min_run)[0]   # windows that are fully above threshold

    if len(valid_starts):
        return int(valid_starts[-1] + min_run - 1)   # right edge of the last fully-signal run
    return len(q) - 1

