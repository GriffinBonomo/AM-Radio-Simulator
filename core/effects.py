import numpy as np
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

def add_crackle(signal: np.ndarray, sample_rate: int, density=0.001, intensity=0.3):
    crackle_track = np.zeros_like(signal) # just the crackles
    num_crackles = int(len(signal) * density)
    indices = np.random.randint(0, len(signal), num_crackles)

    for i in indices:
        burst_length = np.random.randint(int(0.0005 * sample_rate), int(0.002 * sample_rate))
        end = min(i + burst_length, len(signal))
        if end <= i:
            continue

        burst = np.random.randn(burst_length) * intensity
        crackle_track[i:end] += burst[:end - i]

    result = signal + crackle_track
    return np.clip(result, -1, 1)

