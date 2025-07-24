# src/bci_core/feature_extractor.py

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
        eeg_bands (dict): EEG bands as (low_freq, high_freq) tuples.
        bandpower (dict): bandpower[channel][band] → {absolute, relative}
    """

    def __init__(self, sfreq: float = 250.0):
        """
        Initialise the band definitions and internal storage.

        Args:
            sfreq (float): Sampling frequency in Hz (default: 250.0)
        """
        self.sfreq = sfreq

        # Define full EEG band dictionary (standard + extended bands)
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'mu': (8, 13),
            'low_alpha': (8, 10),
            'high_alpha': (10, 12),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45),
            'high_gamma': (60, 100)
        }

        self.bandpower = {}  # Will be populated after feature extraction

    def extract_features(self, signal: np.ndarray):
        """
        Compute absolute and relative bandpower for each EEG band and channel.

        Uses Welch’s method to estimate power spectral density (PSD),
        then integrates PSD over each frequency band using NumPy’s trapezoid rule.

        Args:
            signal (np.ndarray): EEG data of shape (n_channels, n_samples)

        Raises:
            ValueError: If input is not 2D.
        """
        
        if signal.ndim != 2:
            raise ValueError("Signal must be a 2D array (n_channels × n_samples)")

        n_channels = signal.shape[0]
        self.bandpower = {ch: {} for ch in range(n_channels)}

        for ch in range(n_channels):
            # Welch PSD estimation for channel
            freqs, psd = welch(signal[ch], fs=self.sfreq)

            # Total power in full frequency range (used for relative computation)
            total_power = np.trapezoid(psd, freqs)

            for band, (low_f, high_f) in self.eeg_bands.items():
                # Frequency indices within band
                idx_band = np.logical_and(freqs >= low_f, freqs <= high_f)

                # Absolute band power via trapezoidal integration
                band_power = np.trapezoid(psd[idx_band], freqs[idx_band])

                # Normalised relative band power (fraction of total)
                rel_power = band_power / total_power if total_power > 0 else 0.0

                # Store in nested dictionary
                self.bandpower[ch][band] = {
                    'absolute': band_power,
                    'relative': rel_power
                }

    def get_bandpower(self, channel: int, band: str, mode: str = 'absolute') -> float:
        """
        Retrieve bandpower for a given channel and band.

        Args:
            channel (int): EEG channel index (0-based)
            band (str): Band name (e.g. 'alpha', 'mu')
            mode (str): 'absolute' or 'relative'

        Returns:
            float: Bandpower value for that channel and band

        Raises:
            ValueError: If mode, band, or channel is invalid
        """
        if band not in self.eeg_bands:
            raise ValueError(f"Unknown band: {band}")
        if channel not in self.bandpower:
            raise ValueError(f"Channel {channel} not found. Run extract_features() first.")
        if mode not in ['absolute', 'relative']:
            raise ValueError(f"Mode must be 'absolute' or 'relative'")

        return self.bandpower[channel][band][mode]
