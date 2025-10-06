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
