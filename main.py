import argparse
import numpy as np
from pydub import AudioSegment
from scipy.signal import butter, lfilter


def load_audio_to_numpy(path: str) -> tuple[np.ndarray, int]:
    """Load audio into NumPy array, return samples + sample rate."""
    audio = AudioSegment.from_file(path)
    samples = np.array(audio.get_array_of_samples())

    # Convert to mono if stereo
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(axis=1).astype(samples.dtype)

    return samples, audio.frame_rate


def normalize(samples: np.ndarray) -> np.ndarray:
    """Normalize samples to [-1, 1]."""
    return samples.astype(np.float32) / np.max(np.abs(samples))


def am_modulate(samples: np.ndarray, sample_rate: int, carrier_frequency: int) -> np.ndarray:
    """Amplitude modulate the audio onto a carrier wave."""
    offset: int = 1 # Lowering this below 1 causes the distortion heard while tuning a radio.

    time_interval = np.arange(len(samples)) / sample_rate
    carrier = np.cos(2 * np.pi * carrier_frequency * time_interval)
    return (offset + samples) * carrier


def add_noise(signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to simulate static."""
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise


def butter_lowpass(cutoff, sample_rate, order=6):
    """Design a Butterworth low-pass filter."""
    nyquist_frequency = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist_frequency

    # Parameters for a lowpass filter, to be passed into lfilter.
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def am_envelope_demodulate(signal: np.ndarray, sample_rate: int, cutoff: int, full_wave: bool = True) -> np.ndarray:
    """Demodulate AM signal using rectification + low-pass Butterworth filter."""
    order: int = 6

    if full_wave:
        # Use full-wave rectification
        rectified = np.abs(signal)
    else:
        # Use half-wave rectification (old radio style)
        rectified = np.maximum(0, signal)

    b, a = butter_lowpass(cutoff, sample_rate, order)
    return lfilter(b, a, rectified)

def am_synchronous_demodulate(signal: np.ndarray, sample_rate: int, carrier_frequency: int, cutoff: int) -> np.ndarray:
    time_interval = np.arange(len(signal)) / sample_rate
    # Local oscillator at carrier frequency
    local_osc = np.cos(2 * np.pi * carrier_frequency * time_interval)

    # Multiply (mix) signal with local oscillator
    mixed = signal * local_osc

    # Low-pass filter to remove 2*fc component and leave baseband
    b, a = butter_lowpass(cutoff, sample_rate, order=6)
    demod = lfilter(b, a, mixed)

    return demod

def save_wav(samples: np.ndarray, sample_rate: int, path: str):
    """Save a NumPy array as a WAV file."""
    samples = samples / np.max(np.abs(samples))  # normalize
    samples = (samples * 32767).astype(np.int16)

    out = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=samples.dtype.itemsize,
        channels=1
    )
    out.export(path, format="wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input audio file (.wav or .mp3)")
    args = parser.parse_args()

    carrier_frequency: int = 10000
    lowpass_cutoff: int = 5000
    envelope_demod: bool = False

    # Load + normalize
    samples, sample_rate = load_audio_to_numpy(args.input_file)
    normalized_samples = normalize(samples)

    # Modulate
    am_signal = am_modulate(normalized_samples, sample_rate, carrier_frequency)

    # Add static noise
    received = add_noise(am_signal, noise_level=0.05)

    # Demodulate
    if envelope_demod:
        demodulated = am_envelope_demodulate(received, sample_rate, lowpass_cutoff) # Carrier frequency still audible
    else:
        demodulated = am_synchronous_demodulate(received, sample_rate, carrier_frequency, lowpass_cutoff) # Cleaner

    # Save output
    save_wav(demodulated, sample_rate, "output.wav")
    print("Saved AM demodulated audio as output.wav")


if __name__ == "__main__":
    main()
