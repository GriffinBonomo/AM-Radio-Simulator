from scipy.signal import butter, lfilter

def butter_filter(signal, sample_rate, cutoff, btype, order=6):
    """Butterworth filter. btype accepts 'high' and 'low' for a highpass and lowpass filter respectively."""
    nyquist_frequency = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist_frequency

    # Parameters for a lowpass filter, to be passed into lfilter.
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, signal)
