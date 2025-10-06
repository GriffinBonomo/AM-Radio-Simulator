import numpy as np

def normalize(samples: np.ndarray) -> np.ndarray:
    """Normalize samples to [-1, 1]."""
    return samples.astype(np.float32) / np.max(np.abs(samples))
