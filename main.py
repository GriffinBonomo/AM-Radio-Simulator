import argparse
import numpy as np
from pydub import AudioSegment


def load_audio_to_numpy(path: str) -> np.ndarray:
    """
    Load a WAV or MP3 file into a NumPy array.
    """
    # Pydub loads into AudioSegment
    audio = AudioSegment.from_file(path)

    # Get raw samples as array of integers
    samples = np.array(audio.get_array_of_samples())

    # If stereo, reshape into (n_samples, n_channels)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input audio file (.wav or .mp3)")
    args = parser.parse_args()

    samples = load_audio_to_numpy(args.input_file)
    print(f"Loaded {args.input_file}")
    print(f"Shape: {samples.shape}, dtype: {samples.dtype}")


if __name__ == "__main__":
    main()
