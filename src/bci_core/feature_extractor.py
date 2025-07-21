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
                band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
                band_powers.append(band_power)

            # Store as NumPy array.
            bandpower_dict[band] = np.array(band_powers)

        return bandpower_dict
    
    def compute_relative_bandpower(self, signal: np.ndarray, fs: int):
        """
        Method: compute_relative_bandpower

        Goal:
            Calculate the relative power (i.e., normalised power) in each EEG frequency band.
            This is computed as the proportion of total bandpower per channel.

        Parameters:
            signal (np.ndarray): EEG data of shape (n_channels, n_samples).
            fs (int): Sampling rate in Hz.

        Returns:
            dict[str, np.ndarray]:
                - Keys are frequency band names.
                - Values are 1D NumPy arrays of relative power (length = n_channels), all 
                between 0 and 1.

        Clinical + Signal Processing Rationale:
            - Absolute bandpower gives the raw power within a band (e.g., μV²).
            - Relative bandpower expresses this as a proportion of total power across all 
            bands, per channel.
            - Clinically, this helps compare signal features across subjects with different 
            baseline amplitudes (e.g. drowsiness or mental workload monitoring).
        """
        # 1.- Calculate absolute power for each band.
        absolute_power = self.compute_absolute_bandpower(signal, fs)

        # 2.- Stack all absolute powers into a 2D array: shape = (n_bands, n_channels),
        #     all between 0 and 1.

        # 3.- Compute total power per channel (e.g, column-wise sum).
        #     This  gives us: total_power = [P_ch1, P_ch2, ..., P_chN].

        # 4.- Avoid divide-by-zero: add small ε total_power is 0 anywhere.

        # 5.- Compute relative power per band: divide each row by total_power.
        #     Result is: relative_matrix[band, ch] = absolute_power[band, ch].

        # 6.- Convert back to dictionary: key = band name, value = relative power per channel.
