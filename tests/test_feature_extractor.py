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
# FIXTURE: Simulate 4-channel EEG (4s @ 250 Hz):
# -----------------------------

@pytest.fixture
def synthetic_eeg_signal():
    np.random.seed(42)
    n_channels = 4
    duration_sec = 4
    sampling_rate = 250
    signal_length = duration_sec * sampling_rate
    signal = np.random.randn(n_channels, signal_length)
    return signal, sampling_rate

# -----------------------------
# TEST: Class instantiation:
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
# TEST: Absolute Bandpower Output:
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
    fe = FeatureExtractor()

    bandpower = fe.compute_absolute_bandpower(signal, fs)

    expected_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Ensure output is a dictionary with exactly 5 EEG bands.
    assert isinstance(bandpower, dict), 'Output must be a dictionary.'
    assert set(bandpower.keys()) == set(expected_bands), 'Missing or extra EEG bands in results.'

    # Validate each band's power output (5 of them).
    for band in expected_bands:
        values = bandpower[band]

        # Output must be a NumPy array.
        assert isinstance(values, np.ndarray)
        
        # Must be 1D: one value per channel.
        assert values.ndim == 1, f'{band} bandpower must be 1D.'

        # Length must equal number of input channels.
        assert values.shape[0] == signal.shape[0], f'{band} length mismatch with input channels.'

        # All values must be >= 0 (power cannot be negative).
        assert np.all(values >= 0), f'{band} bandporwer contains negative power values.'

# -----------------------------
# TEST: Relative Bandpower Output:
# -----------------------------

def test_relative_bandpower_output(synthetic_eeg_signal):
    """
    Relative Bandpower:
                Goal:
        - Confirm that each EEG band returns a 1D NumPy array (length = n_channels).
        - All values are between 0 and 1 (inclusive).
        - For each channel, the total relative power across bands should ≈ 1.
    """
    signal, fs = synthetic_eeg_signal
    fe = FeatureExtractor()

    rel_power = fe.compute_relative_bandpower(signal, fs)
    expected_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Output must be a dictionary with expected EEG band keys.
    assert isinstance(rel_power, dict), 'Output must be a dictionary.'
    assert set(rel_power.keys()) == set(expected_bands), 'Band keys mismatch.'

    # Shape, range, and summation checks.
    channel_count = signal.shape[0] # shape[0]: number of rows.
    band_matrix = [] 

    for band in expected_bands:
        band_vals = rel_power[band]

        # Must be a 1D NumPy array.
        assert isinstance(band_vals, np.ndarray), f'{band} must return a NumPy array.'
        assert band_vals.ndim == 1, f'{band} output must be 1D.'
        assert band_vals.shape[0] == channel_count, f'{band} length must match number of channels.'

        # Must lie within [0, 1].
        assert np.all(band_vals >= 0) and np.all(band_vals <= 1), f'{band} contains out-of-range values.'

        band_matrix.append(band_vals)

    # Ensure sum of all relative powers across bands = 1 per channel (within tolerance).
    total_per_channel = np.sum(band_matrix, axis=0)
    np.testing.assert_allclose(total_per_channel, np.ones(channel_count), atol=1e-6)