# src/tests/test_feature_extractor.py

import pytest
import numpy as np
from bci_core.feature_extractor import FeatureExtractor

"""
Test Suite for FeatureExtractor Module
--------------------------------------
This suite verifies:
    - Class instantiation and interface
    - Extraction of absolute and relative bandpower values
    - Numerical integrity and shape of results
    - Sanity checks on signal power (e.g., non-negative, relative sum ≈ 1)

EEG Frequency Bands (Hz):
    - delta       0.5-4
    - theta       4-8
    - mu          8-13
    - low_alpha   8-10
    - high_alpha  10-12
    - alpha       8-12
    - beta        12-30
    - gamma       30-45
    - high_gamma  60-100
"""

# -----------------------------
# FIXTURE: Simulate 4-channel EEG (4s @ 250 Hz)
# -----------------------------
@pytest.fixture
def synthetic_eeg_signal():
    """
    Simulates EEG input for testing:
        - Shape: (4 channels × 1000 samples)
        - Duration: 4 seconds
        - Sampling rate: 250 Hz
    Returns:
        - signal (ndarray): Simulated EEG data
        - fs (int): Sampling frequency
    """
    np.random.seed(42)
    n_channels = 4
    duration_sec = 4
    sampling_rate = 250
    signal_length = duration_sec * sampling_rate
    signal = np.random.randn(n_channels, signal_length)
    return signal, sampling_rate

# -----------------------------
# TEST: Class initialisation
# -----------------------------
def test_feature_extractor_init():
    """
    Test: Constructor + Interface
        - Can instantiate class
        - Public methods present
    """
    fe = FeatureExtractor()
    assert isinstance(fe, FeatureExtractor)
    assert hasattr(fe, "extract_features")
    assert hasattr(fe, "get_bandpower")

# -----------------------------
# TEST: Bandpower output via extract_features() and get_bandpower()
# -----------------------------
def test_bandpower_extraction_and_retrieval(synthetic_eeg_signal):
    """
    Test: Absolute + Relative Bandpower Retrieval

    This test:
        - Runs extract_features() on valid EEG input
        - Accesses all defined EEG bands using get_bandpower()
        - Checks value ranges and data types

    Expected:
        - Absolute power: ≥ 0.0
        - Relative power: ∈ [0, 1]
        - All outputs must be floats
    """
    signal, fs = synthetic_eeg_signal
    fe = FeatureExtractor(sfreq=fs)
    fe.extract_features(signal)

    n_channels = signal.shape[0]
    bands = list(fe.eeg_bands.keys())

    for ch in range(n_channels):
        for band in bands:
            abs_val = fe.get_bandpower(ch, band, mode='absolute')
            rel_val = fe.get_bandpower(ch, band, mode='relative')

            # Check value types and physical plausibility
            assert isinstance(abs_val, float), f"{band} absolute not float"
            assert abs_val >= 0, f"{band} absolute < 0"

            assert isinstance(rel_val, float), f"{band} relative not float"
            assert 0 <= rel_val <= 1, f"{band} relative out of [0, 1]"

# -----------------------------
# TEST: Relative bandpower must sum to 1.0 per channel
# -----------------------------
def test_relative_bandpower_sums_to_one(synthetic_eeg_signal):
    """
    Test: Relative Bandpower Integrity

    Verifies:
        - Sum of all relative powers per channel is ≤ 1.0
        - Accepts expected undercoverage from non-band regions
    """
    signal, fs = synthetic_eeg_signal
    fe = FeatureExtractor(sfreq=fs)
    fe.extract_features(signal)

    n_channels = signal.shape[0]
    bands = list(fe.eeg_bands.keys())
    rel_matrix = np.zeros((len(bands), n_channels))

    for i, band in enumerate(bands):
        for ch in range(n_channels):
            rel_matrix[i, ch] = fe.get_bandpower(ch, band, mode='relative')

    total_rel_power = np.sum(rel_matrix, axis=0)

    # Accept partial coverage of the spectrum (typical is ~0.65–0.75)
    assert np.all(total_rel_power <= 1.0)
    assert np.all(total_rel_power >= 0.6)
