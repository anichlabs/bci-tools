# tests/test_preprocessor.py

# -----------------------------------------------------------------------
# Unit tests for SignalPreprocessor:
#
# This module verifies the correctness and robustness of the EEG signal
# preprocessing routines, including filtering and normalisation.
#
# Each test reflects common EEG scenarios and edge cases:
#   - Single-channel (1D) and multi-channel (2D) signals
#   - Presence of noise
#   - Shape preservation
#   - Type safety and error handling
# -----------------------------------------------------------------------

# Import the required test tools and Numpy.
import pytest
import numpy as np
from bci_core.preprocessor import SignalPreprocessor # Import the class to test from the BCI core package.

# ---------------------------
# FIXTURES: Simulated signals
# ---------------------------

"""
@pytest.fixture: This decorator marks the function as a reusable fixture 
for unit tests in pytest.
Why? Fixtures promote clean, modular tests. Instead of duplicating the 
signal generation logic in every test, define it once and inject it where
needed.
"""
@pytest.fixture
def synthetic_1d_signal():
    """
    Simulated 1D EEG signal:
        - Duration: 4 seconds
        - Frequency: 10 Hz sine wave
        - Noise: Added Gaussian noise (mean=0, std=0.5)
          (Noise values are distributed in a bell-shaped curve with a specific 
          mean and standard deviation. It's commonly encountered in various 
          fields like signal processing, communications, and image processing.)
    Condiderations:
        250 Hz is a very common sampling rate in real EEG devices, including 
        BrainFlow/OpenBCI.
        It gives enough resolution to capture important bands (delta to gamma) 
        without oversampling.


    Returns:
        np.ndarray of shape (1000,), dtype=float64
    """
    fs = 250 # Sampling frequency.
    """
    Correct Usage of np.linspace:
        - start=0: begin at time = 0 seconds
        - stop=4: end at 4 seconds
        - num=4 * fs: generate 1000 samples (with fs=250)
        - endpoint=False: exclude the stop value (used to avoid aliasing issues)
    This creates a time vector that simulates 4 seconds of EEG sampled at 
    250 Hz exactly what is needed for signal synthesis.
    """
    # Generate the time vector for the signal.
    t = np.linspace(0, 4, 4 * fs, endpoint=False) # 250 samples per second during 4 seconds (1000 spaced samples).
    """
    PURE:
    Create the core sinusoidal signal at 10 Hz:
        - 2πf: converts frequency (Hz) into angular frequency (radians per second).
        - t: the time vector.
    Result: a clean, band-limited waveform (like an alpha brainwave).

    NOISY:
    Adds realistic Gaussian noise to the signal:
        - np.random.randn(len(t)): draws 1000 samples from a normal distribution 
          (mean=0, std=1).
        - 0.5 * ...: scales the standard deviation to 0.5 µV (microvolts), 
          mimicking mild artefacts or sensor noise.

    Why add noise?
        Real EEG is never noise-free.
        Test robustness of filters (like bandpass) against realistic disturbances.
    """
    pure = np.sin(2 * np.pi * 10 * t) # Pure alpha wave (10 Hz), (which occurs naturally in relaxed wakefulness).
    noisy = pure + 0.5 * np.random.randn(len(t))
    return noisy

@pytest.fixture
def synthetic_2d_signal():
    """
    Simulated 2D EEG signal:
        - Channels: 4
        - Duration: 4 seconds
        - Each channel: different base frequency + noise
    Returns:
        np.ndarray of shape (4, 1000)
    """
    fs = 250
    t = np.linspace(0, 4, 4 * fs, endpoint=False)
    signals = []
    # Simulated EEG subbands. Cover multiple bands (Theta, Alpha, Beta, Gamma).
    for f in [5, 10, 20, 40]:
        base = np.sin(2 * np.pi * f * t)
        noisy = base + 0.3 * np.random.randn(len(t))
        signals.append(noisy)
    return np.array(signals)

# --------------------------------
# TESTS: BANDPASS FILTER FUNCTION
# --------------------------------

def test_bandpass_filter_1d(synthetic_1d_signal):
    """
    Test: Filtering a 1D EEG signal should preserve shape,
    reduce out-of-band noise, and not crash on realistic input.
    """
    pre = SignalPreprocessor(lowcut=8, highcut=12, fs=250, order=4)
    filtered = pre.bandpass_filter(synthetic_1d_signal)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == synthetic_1d_signal.shape
    assert not np.any(np.isnan(filtered))

def test_bandpass_filter_2d(synthetic_2d_signal):
    """
    Test: Filtering a 2D EEG array should preserve shape (channels, timepoints),
    and apply filter to each channel independently.
    """
    pre = SignalPreprocessor(lowcut=5, highcut=45, fs=250)
    filtered = pre.bandpass_filter(synthetic_2d_signal)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == synthetic_2d_signal.shape
    assert filtered.ndim == 2
    assert not np.any(np.isnan(filtered))

def test_bandpass_invalid_dim():
    """
    Test: Invalid input dimensionality (e.g. 3D EEG or object) 
    should raise ValueError.
    """
    pre = SignalPreprocessor()
    bad_data = np.zeros((2, 2, 2)) # Invalid shape for EEG.

    with pytest.raises(ValueError):
        pre.bandpass_filter(bad_data)
    
# -------------------------------------
# TESTS: Z-SCORE NORMALISATION FUNCTION
# -------------------------------------

def test_z_score_1d_normalisation(synthetic_1d_signal):
    """
    Test: Z-score normalisation on 1D EEG signal should:
        - Return zero mean.
        - Return unit variance.
        - Preserve shape.
    """
    pre = SignalPreprocessor()
    normed = pre.z_score_normalise(synthetic_1d_signal)

    assert isinstance(normed, np.ndarray)
    assert normed.shape == synthetic_1d_signal.shape
    # assert_allclose: comparison between 2 objects to see if they are equal.
    """
    Whenever any computation is performed using floating point numbers, chances 
    are very high that result is somewhat different from actual theoretical 
    value. This is because of the floating point arithmetic.

    To compare our result with the theoretical result, we need to do so with 
    some tolerance for the error. Although, bigger our tolerance, bigger is 
    the chance of getting something very erroneous marked as correct.
    """
    np.testing.assert_allclose(np.mean(normed), 0, atol=1e-7) # rtol: Relative Tolerance.
                                                              # atol: Absolute Tolerance.
    np.testing.assert_allclose(np.std(normed), 1, atol=1e-7)

def test_z_score_2d_normalisation(synthetic_2d_signal):
    """
    Test: Per-channel Z-score normalisation should result in:
        - Each channel having mean ≈ 0
        - Each channel having std ≈ 1
        - Same shape as input
    """
    pre = SignalPreprocessor()
    normed = pre.z_score_normalise(synthetic_2d_signal)

    assert isinstance(normed, np.ndarray)
    assert normed.shape == synthetic_2d_signal.shape
    assert not np.any(np.isnan(normed)) # Expect NO NaNs.
    for channel in normed:
        np.testing.assert_allclose(np.mean(channel), 0, atol=0.03)
        np.testing.assert_allclose(np.std(channel), 1, atol=0.03)