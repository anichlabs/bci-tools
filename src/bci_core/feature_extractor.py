# src/feature_extractor.py

import numpy as np

class FeatureExtractor:
    """
    FeatureExtractor class for EEG signal analysis.

    Provides methods to compute absolute and relative power
    for standard EEG frequency bands: delta, theta, alpha, beta, gamma.
    """

    def compute_absolute_bandpower(self, signal: np.ndarray):
        """
        Placeholder method for absolute bandpower computation.
        
        Parameters:
            signal (np.ndarray): EEG data (shape: channels x samples)
            fs (int): Sampling frequency in Hz

        Returns:
            dict: Keys are EEG bands, values are arrays of power per channel
        """
        raise NotImplementedError('compute_absolute_bandpower() not implemented yet.')