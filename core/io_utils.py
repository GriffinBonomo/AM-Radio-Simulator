import numpy as np
from pydub import AudioSegment

def load_audio_to_numpy(path: str) -> tuple[np.ndarray, int]:
    """Load audio into NumPy array, return samples + sample rate."""
    audio = AudioSegment.from_file(path)
    samples = np.array(audio.get_array_of_samples())

    # Convert to mono if stereo
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(axis=1).astype(samples.dtype)

    return samples, audio.frame_rate

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