import argparse
from config import *
from core.io_utils import load_audio_to_numpy, save_wav
from core.utils import normalize
from core.modulation import am_modulate, am_synchronous_demodulate
from core.effects import add_noise_bandlimited, add_noise_fullspectrum, add_crackle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input audio file (.wav or .mp3)")
    args = parser.parse_args()

    # Load samples
    samples, sample_rate = load_audio_to_numpy(args.input_file)
    # Normalize
    normalized_samples = normalize(samples)
    # Modulate
    am_signal = am_modulate(normalized_samples, sample_rate, CARRIER_FREQUENCY)

    # Add effects
    received = add_noise_bandlimited(am_signal, sample_rate)
    received = add_crackle(received, sample_rate, CRACKLE_DENSITY, CRACKLE_INTENSITY)

    # De-modulate
    demodulated = am_synchronous_demodulate(received, sample_rate, CARRIER_FREQUENCY, LOWPASS_CUTOFF)
    
    # Save as .WAV
    save_wav(demodulated, sample_rate, "output.wav")
    print("Saved AM demodulated audio as output.wav")

if __name__ == "__main__":
    main()
