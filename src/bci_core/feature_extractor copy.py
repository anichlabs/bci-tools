# src/feature_extractor.py

import numpy as np
from scipy.signal import welch

class FeatureExtractor:
    """
    EEG Feature Extraction Module.

    This class extracts absolute and relative power features from EEG signals
    across a wide range of standard and extended frequency bands, including:

        - Delta       (0.5-4 Hz)
        - Theta       (4-8 Hz)
        - Mu          (8-13 Hz)
        - Low Alpha   (8-10 Hz)
        - High Alpha  (10-12 Hz)
        - Alpha       (8-12 Hz)
        - Beta        (12-30 Hz)
        - Gamma       (30-45 Hz)
        - High Gamma  (60-100 Hz)
    
    Power values are calculated using Welch's method (non-parametric PSD estimate),
    and stored for later access per channel and frequency band.

    Attributes:
        sfreq (float): Sampling frequency in Hz.
        bands (dict): Dictionary of EEG frequency bands with (low, high) Hz.
        bandpower (dict): Nested structure: bandpower[channel][band] → value
    """

    def __init__(self, sfreq: float = 250.0):
        """
        Initialise standard EEG frequency bands (in Hz).
        These ranges are based on clinical convention.
        """
        self.sfreq = sfreq

        # Full set of standard + extended EEG bands.
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'mu': (8, 10),
            'low_alpha': (8, 10),
            'high_alpha': (8, 12),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45),
            'high_gamma': (60, 100)
        }

        # Storage for computed bandpowers.
        self.bandpower = {}

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
        band_matrix = np.stack([absolute_power[band] for band in self.eeg_bands.keys()], axis=0)
        
        # 3.- Compute total power per channel (e.g, column-wise sum).
        #     This  gives us: total_power = [P_ch1, P_ch2, ..., P_chN].
        total_power = np.sum(band_matrix, axis=0) # axis=0 refers to row.

        # 4.- Avoid divide-by-zero: add small ε total_power is 0 anywhere.
        total_power = np.where(total_power == 0, 1e-12, total_power)

        # 5.- Compute relative power per band: divide each row by total_power.
        #     Result is: relative_matrix[band, ch] = absolute_power[band, ch].
        relative_matrix = band_matrix / total_power

        # 6.- Convert back to dictionary: key = band name, value = relative power per channel.
        relative_power = {
            band:relative_matrix[i, :]
            for i, band in enumerate(self.eeg_bands.keys())
        }

        return relative_power
    
    