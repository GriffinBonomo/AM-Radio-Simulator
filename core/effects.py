import numpy as np
from scipy.signal import butter, lfilter
from .filters import butter_filter

def add_noise_fullspectrum(signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to simulate static."""
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise

def add_noise_bandlimited(signal: np.ndarray, sample_rate: int, noise_level: float = 0.05, low_cutoff: int = 3000, high_cutoff: int = 10000, order: int = 6) -> np.ndarray:
    """Add band-limited noise to simulate radio hiss"""
    noise = noise_level * np.random.randn(len(signal))

    filtered = butter_filter(noise, sample_rate, low_cutoff, "high", order)
    filtered = butter_filter(filtered, sample_rate, high_cutoff, "low", order)
    return signal + filtered

def add_crackle(signal: np.ndarray, sample_rate: int, density: float, intensity: float):
    crackle_track = np.copy(signal)
    num_crackles = int(len(signal) * density)
    indices = np.random.randint(0, len(signal), num_crackles)

    for i in indices:
        burst_length = np.random.randint(int(0.0005 * sample_rate), int(0.002 * sample_rate))
        end = min(i + burst_length, len(signal))
        if end <= i:
            continue

        burst = np.random.randn(burst_length)

        b, a = butter(2, [0.05, 0.3], btype='band') 
        burst = lfilter(b, a, burst)
        burst *= intensity

        crackle_track[i:end] += burst[:end - 1]

    return np.clip(crackle_track, -1, 1)

