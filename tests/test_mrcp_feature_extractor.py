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
    mimics a downward-trending MRCP deflection.

    This synthetic EEG trial has each channel linearly decreasing
    from 0.0 μV at -4.0 seconds to -10.0 μV at 0.0 seconds. This simulates
    the slow negativity buildup typical of pre-movement MRCPs.

    The area under such a signal (trapezoid rule) should be negative.
    '''

    
