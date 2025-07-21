# tests/test_feature_extractor.py

import pytest
import numpy as np
from bci_core.feature_extractor import FeatureExtractor

"""
Scaffold:
This first test suite will:
    - Instantiate the class.
    - Input a synthetic signal: shape (4 channels x 1000 samples) (equivalent to 4s @ 250 Hz)
    - Check the output of each feature method
Include:
    - Absolute bandpower (per EEG band per channel)
    - Relative bandpower (same as above, normalised)
    - Output shape & numerical sanity checks
EEG Frequency Bands (Hz)
    - Assume standard clinical ranges:
        Band	Range (Hz)
        Delta	0.5-4
        Theta	4-8
        Alpha	8-13
        Beta	13-30
        Gamma	30-45
"""

# -----------------------------
# FIXTURE: Simulate 4-channel EEG (4s @ 250 Hz)
# -----------------------------

@pytest.fixture
def synthetic_egg_signal():
    np.random.seed(42)
    n_channels = 4
    duration_sec = 4
    sampling_rate = 250
    signal_length = duration_sec * sampling_rate
    signal = np.random.randn(n_channels, signal_length)
    return signal, sampling_rate

# -----------------------------
# TEST: Class instantiation
# -----------------------------

# Instantiate the FeatureExtractor() class.
def test_feature_extractor_init():
    fe = FeatureExtractor
    assert fe is not None

"""
Absolute Bandpower:
This test ensures that:
    - Output is a dict of 5 bands (delta, theta, alpha, beta, gamma).
    - Each entry is a NumPy array with length equal to n_channels.
    - All values are real and non-negative (power cannot be negative).
    - The function behaves as expected given a known input shape.
"""

# -----------------------------
# TEST: Absolute Bandpower Output
# -----------------------------

def test_absolute_bandpower_output(synthetic_eeg_signal):
    """
    Absolute Bandpower:
    Goal:
        - Output is a dict of 5 bands (delta, theta, alpha, beta, gamma).
        - Each entry is a NumPy array with length equal to n_channels.
        - All values are real and non-negative (power cannot be negative).
        - The function behaves as expected given a known input shape.
    """
    signal, fs = synthetic_eeg_signal
    fe = FeatureExtractor

    bandpower = fe.compute_absolute_bandpower(signal, fs)
    
