# test/test_mrcp_feature_extractor.py

import numpy as np
from bci_core.mrcp.feature_extractor import MRCPFeatureExtractor

def test_extract_all_returns_correct_keys():
    '''
    Test whether MRCPFeatureExtractor.extract_all() returns a valid
    nested dictionary with keys 'area', 'peak', and 'slope' per channel.

    This test uses synthetic EEG data (4 channels x 1000 samples)
    simulating a 4-second trial sampled at 250 Hz. The last sample
    corresponds to t = 0.0 (movement onset).

    The test asserts that:
        - All channels are processed.
        - Each channel returns all three features.
    '''
    # Generate a synthetic EEG trial.
    # 4 channels, 1000 samples (4 seconds at 250 Hz).
    eeg = np.random.randn(4, 1000) * 5e-6 # Simulate low-amplitud EEG in μV range.
    
    # Instantiate the extractor.
    mrcp = MRCPFeatureExtractor(sfreq=250.0)

    # Extract all features.
    features = mrcp.extract_all(eeg)

    # Validate top-level keys (channels).
    assert len(features) == 4

    for ch in features:
        # Ensure each channel contains all expected metrics.
        assert 'area' in features[ch]
        assert 'peak' in features[ch]
        assert 'slope' in features[ch]

def test_area_is_negative_for_downward_mrcp():
    '''
    Test that extract_area() returns negative values when input signal
    mimics a downward-trending MRCP deflection, which is known as 
    'Bereitschaftspotential.'

    This synthetic EEG trial has each channel linearly decreasing
    from 0.0 μV at -4.0 seconds to -10.0 μV at 0.0 seconds. This simulates
    the slow negativity buildup typical of pre-movement MRCPs.

    The area under such a signal (trapezoid rule) should be negative.
    '''
    sfreq = 250 # Sampling frequency in Hz.
    duration_sec = 4.0
    n_samples = int(duration_sec * sfreq)

    # Time vector: -4.0 to 0.0 seconds.
    t = np.linspace(-duration_sec, 0, n_samples)

    # Create a linear negative ramp per channel.
    n_channels = 4
    eeg = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        eeg[ch] = -10e6 * (t / t[-1]) # Linear from 0 to -10 μV.

    # Instantiate and extract.
    mrcp = MRCPFeatureExtractor(sfreq=sfreq)
    areas = mrcp.extract(eeg)

    # Assert all areas are negative.
    for ch in range(n_channels):
        assert areas[ch] < 0.0

def test_peak_returns_minimum_in_window():
    '''
    Test that extract_peak() correctly identifies the most negative
    value in the interval [-1.5, 0.0] seconds.

    This synthetic signal contains a clear negative peak (-25-μV)
    inserted at -0.3 seconds, which is inside the MRCP window.

    The test ensures that this minimum is detected correctly by
    the peak extractor for each channel.
    '''
    sfreq = 250
    duration_sec = 4.0
    n_samples = int(duration_sec * sfreq)
    t = np.linspace(-duration_sec, 0, n_samples)

    n_channels = 4
    eeg = np.random.normal(loc=0.0, scale=1e-6, size=(n_channels, n_samples))

    # Inject a known negative peak at -0.3 seconds.
    idx_peak = np.argmin(np.abs(t = 0.3)) # Find index closest to t = -0.3.

    for ch in range(ch_channels):
        egg[ch, idx_peak] = -25e-6 # Set sharp minimum.
    
    # Instantiate and extract.
    mrcp = MRCPFeatureExtractor(sfreq=sfreq)
    peaks = mrcp.extract_peaks(eeg)

    # Assert that each detected peak equals the injected -25 μV.
    for ch in range(n_channels):
        assert np.isclose(peaks[ch], -25e-6, atol=1e-7)


    
