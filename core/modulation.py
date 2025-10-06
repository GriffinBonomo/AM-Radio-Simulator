import numpy as np
from .filters import butter_filter

def am_modulate(samples: np.ndarray, sample_rate: int, carrier_frequency: int) -> np.ndarray:
    """Amplitude modulate the audio onto a carrier wave."""
    offset: int = 1 # Lowering this below 1 causes the distortion heard while tuning a radio.

    time_interval = np.arange(len(samples)) / sample_rate
    carrier = np.cos(2 * np.pi * carrier_frequency * time_interval)
    return (offset + samples) * carrier

def am_envelope_demodulate(signal: np.ndarray, sample_rate: int, cutoff: int, full_wave: bool = True) -> np.ndarray:
    """Demodulate AM signal using rectification + low-pass Butterworth filter."""
    order: int = 6

    if full_wave:
        # Use full-wave rectification
        rectified = np.abs(signal)
    else:
        # Use half-wave rectification (old radio style)
        rectified = np.maximum(0, signal)

    return butter_filter(rectified, sample_rate, cutoff, "low", order)

def am_synchronous_demodulate(signal: np.ndarray, sample_rate: int, carrier_frequency: int, cutoff: int) -> np.ndarray:
    time_interval = np.arange(len(signal)) / sample_rate
    # Local oscillator at carrier frequency
    local_osc = np.cos(2 * np.pi * carrier_frequency * time_interval)

    # Multiply (mix) signal with local oscillator
    mixed = signal * local_osc

    return butter_filter(mixed, sample_rate, cutoff, "low")