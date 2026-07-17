import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks, peak_widths, hilbert
from scipy.ndimage import uniform_filter1d

#default lambda for Cu source
LAMBDA_CU_KA = 0.15406

#default test file temporary
DEFAULT_FILE = r'Simulated\NiMo_250_0_TiO2_450_0_Si_0000.dat'

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

def estimate_fringe_frequency(q, R):
    """Rough estimate of the Kiessig-fringe frequency (cycles per
    nm^-1 of q), used only to set the background filter's cutoff below
    (not the reported thickness). A low-order polynomial detrend of
    log10(R) removes the smooth decay so the fringe frequency
    dominates a quick FFT."""
    log_R = np.log10(R)
    trend = np.polyval(np.polyfit(q, log_R, 4), q)
    detrended = (log_R - trend) * np.hamming(len(q))
    dq = np.mean(np.diff(q))
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(detrended), d=dq)

    q_span = q[-1] - q[0]
    valid = freqs > 2.0 / q_span  # exclude residual low-frequency trend leakage
    if not np.any(valid) or spectrum[valid].max() == 0:
        return 10.0 / q_span
    return freqs[valid][np.argmax(spectrum[valid])]

def estimate_fringe_frequency_hilbert(q, R, poly_degree=4):
    """Instantaneous fringe frequency vs q, via the Hilbert transform of
    the same detrended log10(R) signal used by estimate_fringe_frequency.
    Unlike the FFT version (one global scalar, assumes a single stable
    period), this tracks frequency as a function of q - useful when the
    fringe period drifts (graded interfaces) or when several periods
    beat against each other (multilayers), where a single FFT peak would
    misrepresent or blend the underlying periods.
    Returns (inst_freq, envelope), both arrays matching q. `envelope`
    is the analytic-signal amplitude - near the array edges it is
    suppressed by the Hamming window, so phase (and hence inst_freq)
    is unreliable there; callers should trim before summarizing."""
    log_R = np.log10(R)
    trend = np.polyval(np.polyfit(q, log_R, poly_degree), q)
    detrended = (log_R - trend) * np.hamming(len(q))

    analytic = hilbert(detrended)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase, q) / (2 * np.pi)
    envelope = np.abs(analytic)
    return np.abs(inst_freq), envelope


def compare_methods(q, R, edge_trim_fraction=0.08):
    """Runs both estimators over the same (already-cropped) q, R window
    and prints a side-by-side comparison."""
    freq_fft = estimate_fringe_frequency(q, R)
    inst_freq, envelope = estimate_fringe_frequency_hilbert(q, R)

    n = len(inst_freq)
    trim = max(1, int(n * edge_trim_fraction))
    core = slice(trim, n - trim)
    freq_hilbert_median = np.median(inst_freq[core])
    freq_hilbert_std = np.std(inst_freq[core])

    d_fft = 2 * np.pi * freq_fft
    d_hilbert = 2 * np.pi * freq_hilbert_median

    print(f"  FFT global      : freq={freq_fft:.4f} cycles/nm^-1  ->  d~{d_fft:.1f} nm")
    print(f"  Hilbert (median): freq={freq_hilbert_median:.4f} +/- {freq_hilbert_std:.4f}  ->  d~{d_hilbert:.1f} nm")
    print(f"  Hilbert range   : [{inst_freq[core].min():.4f}, {inst_freq[core].max():.4f}]"
          f" (spread wide relative to median => non-constant/beating period)")

    return dict(freq_fft=freq_fft, d_fft=d_fft, inst_freq=inst_freq, envelope=envelope,
                core=core, freq_hilbert_median=freq_hilbert_median, d_hilbert=d_hilbert)


def plot_diagnostics(q, R, edge_idx, q_fit, R_fit, result, title):
    fig, axes = plt.subplots(4, 1, figsize=(8, 11))

    ax = axes[0]
    ax.semilogy(q, R, lw=1)
    ax.axvline(q[edge_idx], color='tab:orange', ls='--', label='critical edge')
    ax.set_ylabel('R')
    ax.set_title(title)
    ax.legend(fontsize=8)

    log_R_fit = np.log10(R_fit)
    trend = np.polyval(np.polyfit(q_fit, log_R_fit, 4), q_fit)
    detrended = (log_R_fit - trend) * np.hamming(len(q_fit))
    ax = axes[1]
    ax.plot(q_fit, detrended, lw=1)
    ax.set_ylabel('detrended log10(R)')
    ax.set_xlabel('q (nm^-1)')

    dq = np.mean(np.diff(q_fit))
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(detrended), d=dq)
    ax = axes[2]
    ax.plot(freqs, spectrum, lw=1)
    ax.axvline(result['freq_fft'], color='tab:red', ls='--',
               label=f"FFT peak = {result['freq_fft']:.3f}")
    ax.set_xlabel('frequency (cycles per q-unit)')
    ax.set_ylabel('|FFT|')
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.plot(q_fit, result['inst_freq'], lw=1, color='tab:gray', label='instantaneous freq (raw)')
    core = result['core']
    ax.plot(q_fit[core], result['inst_freq'][core], lw=1.5, color='tab:blue', label='instantaneous freq (trimmed)')
    ax.axhline(result['freq_fft'], color='tab:red', ls='--', label=f"FFT global = {result['freq_fft']:.3f}")
    ax.axhline(result['freq_hilbert_median'], color='tab:green', ls='--',
               label=f"Hilbert median = {result['freq_hilbert_median']:.3f}")
    ax.set_xlabel('q (nm^-1)')
    ax.set_ylabel('instantaneous freq')
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    q, R = load_reflectivity(DEFAULT_FILE, axis='q_angstrom')

    edge_idx, _ = find_critical_edge(q, R)

    q_fit, R_fit = q[edge_idx:], R[edge_idx:]

    print(DEFAULT_FILE)
    print(f"  usable window: idx {edge_idx} -> end  ({len(q_fit)} points,"
          f" q = {q_fit[0]:.3f} -> {q_fit[-1]:.3f} nm^-1)")

    result = compare_methods(q_fit, R_fit)
    plot_diagnostics(q, R, edge_idx, q_fit, R_fit, result, DEFAULT_FILE)
    plt.show()
