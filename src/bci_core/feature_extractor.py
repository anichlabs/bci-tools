# src/feature_extractor.py

import numpy as np
from scipy.signal import welch

class FeatureExtractor:
    """
    FeatureExtractor class for EEG signal analysis.

    Provides methods to compute absolute and relative power
    for standard EEG frequency bands: delta, theta, alpha, beta, gamma.
    """

    def __init__(self):
        """
        Initialise standard EEG frequency bands (in Hz).
        These ranges are based on clinical convention.
        """        
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    def compute_absolute_bandpower(self, signal: np.ndarray, fs: int):
        """
        Method: compute_absolute_bandpower

        Goal:
            Calculate absolute bandpower in each EEG frequency band using Welch's method.
        
        Parameters:
            signal (np.ndarray): Shape = (n_channels, n_samples) — EEG data.
            fs (int): Sampling frequency in Hz.

        Returns:
            dict[str, np.ndarray]:
                - Each key corresponds to a frequency band.
                - Each value is a 1D array of power values (length = n_channels).
        """
        # Validate input shape.
        if signal.ndim != 2:
            raise ValueError('Signal must be a 2D Numpy aaray (channels x samples)')
        
        n_channels, _ = signal.shape
        bandpower_dict = {}

        # Apply Welch's method to each channel independently.
        for band, (low_f, high_f) in self.eeg_bands.items():
            band_powers = []

            for ch in range(n_channels):
                # Compute Power Spectral Density (PSD) for this channel.
                freqs, psd = welch(signal[ch], fs=fs, nperseg=fs*2)

               # Find indices corresponding to the band range.
                idx_band = np.logical_and(freqs >= low_f, freqs <= high_f)

                # Integrate PSD over the band to get absolute power.
                band_power = np.trapz(psd[idx_band], freqs[idx_band])
                band_powers.append(band_power)

            # Store as NumPy array.
            bandpower_dict[band] = np.array(band_powers)

        return bandpower_dict