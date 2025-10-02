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


def am_modulate(samples: np.ndarray, fs: int, fc: int = 10000) -> np.ndarray:
    """Amplitude modulate the audio onto a carrier wave."""
    t = np.arange(len(samples)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    return (1 + samples) * carrier


def add_noise(signal: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to simulate static."""
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise


def butter_lowpass(cutoff, fs, order=6):
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def am_demodulate(signal: np.ndarray, fs: int, cutoff: int = 5000) -> np.ndarray:
    """Demodulate AM signal using rectification + low-pass Butterworth filter."""
    rectified = np.abs(signal)
    b, a = butter_lowpass(cutoff, fs, order=6)
    return lfilter(b, a, rectified)


def save_wav(samples: np.ndarray, fs: int, path: str):
    """Save a NumPy array as a WAV file."""
    samples = samples / np.max(np.abs(samples))  # normalize
    samples = (samples * 32767).astype(np.int16)

    out = AudioSegment(
        samples.tobytes(),
        frame_rate=fs,
        sample_width=samples.dtype.itemsize,
        channels=1
    )
    out.export(path, format="wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input audio file (.wav or .mp3)")
    args = parser.parse_args()

    # Load + normalize
    samples, fs = load_audio_to_numpy(args.input_file)
    norm_samples = normalize(samples)

    # Transmit
    am_signal = am_modulate(norm_samples, fs, fc=10000)

    # Add static noise
    received = add_noise(am_signal, noise_level=0.05)

    # Receive (demodulation with filter)
    demodulated = am_demodulate(received, fs, cutoff=5000)

    # Save demodulated output
    save_wav(demodulated, fs, "output.wav")
    print("Saved AM demodulated audio as output.wav")


if __name__ == "__main__":
    main()
