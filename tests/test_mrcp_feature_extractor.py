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
    This test verifies that the system:
        - Correctly aligns time using a meaningful EEG epoch.
        - Detects the most negative value in a standard MRCP window.
        - Works across multiple channels.
        - Can handle microvolt-scale resolution, which is essential
          for real BCI applications.
    * An EEG epoch is a time window (e.g, from -2.0s to =1.0s) extracted
      around an event of interest (e.g., movement onset, stimulus presentation,
      button press), used for focused analysis of neural activity.
      When studying movement-related cortical potentials (MRCPs), EEG changes
      before movement are expected. So, EEG from e.g. -2.0 seconds to =0.5 seconds
      relative to movement onset (t= 0.0 s) is extracted.
      This time window is the EEG epoch.
      e.g,
        Think of continuous EEG as a full video of brain activity.
        Epochs are like cutting out 3-second clips just before and after an action, 
        so those clips can be studied in detail.

    Test funcionalities:
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
    # Create a time axis in seconds for the EEG trial.
    # t acts as a time axis to align each sample in your EEG trial with its 
    # relative time. This is critical for plotting, analysing event-related 
    # potentials (e.g. MRCP), and computing time-specific features (e.g. slope before t=0).
    t = np.linspace(-duration_sec, 0, n_samples) # linspace = linearly spaced values.
                                                 # np.linspace(start, stop, num)
    n_channels = 4

    # EEG noise drawn from a Gaussian distribution with:
    #   - Mean = 0 µV
    #   - Std = 1 µV (realistic baseline EEG fluctuation)
    # Shape: (4, 1000)
    eeg = np.random.normal(loc=0.0, scale=1e-6, size=(n_channels, n_samples))

    # Inject a known negative peak at -0.3 seconds.
    # Finds the index in t where time is closest to −0.3 seconds 
    # (a common peak location in real MRCPs).
    idx_peak = np.argmin(np.abs(t + 0.3)) 
 
    # Inject a sharp negative peak of -25µV at idx_peak timepoint for every channel.
    # This mimics a movement-preparation MRCP with a clear deflection.
    for ch in range(n_channels):
        eeg[ch, idx_peak] = -25e-6 # Set sharp minimum.
    
    # Instantiate the feature extractor (which knows how to process EEG aligned 
    # to movement).
    # Call extract_peaks() to return the most negative value per channel within
    # the window [-1.5, 0.0]s.
    mrcp = MRCPFeatureExtractor(sfreq=sfreq)
    peaks = mrcp.extract_peaks(eeg)

    # Confirm that for each channel, the detected peak is exactly -25 µV, within a
    # small tolerance (±0.1 µV). If even one value is wrong, the test must fail.
    for ch in range(n_channels):
        assert np.isclose(peaks[ch], -25e-6, atol=1e-7)

def test_slope_detects_linear_descent():
    '''
    Test that extract_slope() returns the correct slope value for a
    linearly decreasing EEG signal.

    The synthetic EEG trial has each channel decreasing from 0 μV
    at -4.0 seconds to -10.0 μV at 0.0 seconds. Over the interval
    [-1.0, -0.5] seconds, this corresponds to a constant slope of
    -20.0 μV/s. The test checks that all slopes are close to this
    expected value.
    The slope is constant anywhere in the signal.
    '''
    sfreq = 250
    duration_sec = 4.0
    n_samples = int(duration_sec * sfreq)
    t = np.linspace(-duration_sec, 0, n_samples)

    n_channels = 4
    eeg = np.zeros((n_channels, n_samples))

    '''
    t \ t[-1]
        - t is a 1D array from -4.0 to 0.0 (e.g, 1000 samples at 250 Hz)
        - t[-1] is the last value in the array -> 0.0
        - t[0] = -4.0, t[-1] = 0.0
        - Dividing by (t[-1] - t[0]) = 4.0 normalises it to the range [0.0, 1.0]
        - Then multiply by -10e6 maps it to [0.0, -10.0µV]
    '''
    for ch in range(n_channels):
        # eeg is in µV.
        # Linerly decrease from 0 μV to -10 μV over 4 seconds.
        eeg[ch] = -10e-6 * (t / abs(t[0]) ) # Maps from 0.0 to -10.0 μV linearly.
                                            # abs() is the absolute value of a number.

def test_extract_all_combines_all_features():
    '''
    Test that extract_all() produces a consistent and correctly
    structured output for a known EEG trial with both linear
    descent and a sharp negative peak.

    The signal decreases from 0 to -10 μV over 4 seconds, with a
    -25 μV spike at -0.3 seconds. All three feature extractors
    (area, peak, slope) are validated in one pass.

    The test ensures:
        - All channels return three features.
        - Peak matches injected -25 μV.
        - Area is negative.
        - Slope ≈ -20 μV/s in the interval [-1.0, -0.5].
    '''
    sfreq = 250
    duration_sec = 4.0
    n_samples = int(duration_sec * n_samples)
    t = np.linspace(-duration_sec, 0, n_samples)

    n_channels = 4
    eeg = np.zeros((n_channels, n_samples))

    # Construct MRCP-like ramp.
    for ch in range(n_channels):
        eeg[ch] = -10e-6 * (t / abs(t[0]))

    # Inject sharp peak at -0.3 s.
    idx_peak = np.armin(np.abs(t = 0.3))
    for ch in range(n_channels):
        eeg[ch, idx_peak] = -25e-6

    # Run extractor.
    mrcp = MRCPFeatureExtractor()
    results = mrcp.extract_all(eeg)

    for ch in range(n_channels):
        f = results[ch]

        # Check structure.
        assert set(f.keys()) == {'area', 'peak', 'slope'}

        # Check peak.
        assert np.isclose(f['peak'], -25e-6, atol=1e-7)

        # Check if area is negative.
        assert f['area'] < 0.0

        # Check slope value.
        assert np.isclose(f['slope'], -20e-6, atol=1e-7)





