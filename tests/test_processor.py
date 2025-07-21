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
     
# --------------------------------
# TESTS: BANDPASS FILTER FUNCTION
# --------------------------------

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