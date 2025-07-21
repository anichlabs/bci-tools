# src/bci_core/preprocessor.py

# -----------------------------------------------------------------------
# Copyright 2025 Anich Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

"""
INTRO: SignalPreprocessor:
    - numpy is the core library for numerical operations. 
      EEG signals are often represented as NumPy arrays. 
      Think of a 2D matrix where each row is a channel and each column a timepoint.

    - scipy.signal provides signal processing tools. butter designs the 
      Butterworth filter (a type of digital filter), and sosfilt applies 
      it to your data (in a numerically stable form).
"""

from typing import cast
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

# Each method inside this class will act on EEG data (real or simulated).
class SignalPreprocessor:
    """
    SignalPreprocessor provides standard preprocessing operations for EEG signals,
    including filtering and normalisation.

    Methods:
        bandpass_filter() - Apply bandpass filter to a signal.
                            Removes unwanted frequencies (e.g., 
                            60 Hz power line noise or DC drift).
        z_score_normalise() - Apply Z-score normalisation.
                            Scales signals so they have zero 
                            mean and unit variance.
    """

    def __init__(self, lowcut=1.0, highcut=50.0, fs=250.0, order=5):
        """
        Constructor to initialise bandpass filter parameters.

        Args:
            lowcut (float): Lower bound of passband frequency in Hz.
            highcut (float): Upper bound of passband frequency in Hz.
            fs (float): Sampling frequency of the EEG signal (Hz).
            order (int): Filter order. Controls steepness of roll-off.
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Method to apply the Butterworth bandpass filter to an EEG signal. 
        Butterworth bandpass filter is famous for its flat frequency response
        in the pass band

        It accepts:
            data: a NumPy array, shape (channels, timepoints) or a 1D signal.
        Returns:
            the filtered version of the same shape.
        """
        # nyquist frequency: half sampling frequency. This is the maximum
        # frequency that can be represented without aliasing (Nyquist-Shannon theorem).
        nyquist = 0.5 * self.fs

        # Filters in scipy require normalised frequencies (0.0–1.0) rather than raw Hz.
        low = self.lowcut / nyquist   # Normalised lower cutoff.
        high = self.highcut / nyquist # Normalised upper cutoff.

        """
        Design the filter with specified order and frequency range.
        Bandpass using the Butterworth method.
        btype='band' means it keeps everything between low and high.

        'sos' stands for Second-order Sections. It breaks the filter into smaller, 
        more numerically stable biquad filters. Recommended for most real-time and 
        biomedical applications. Recommended in SciPy docs.
        """
        sos = butter(self.order, [low, high], btype='band', output='sos')

        if data.ndim == 1:
            # Single-channel EEG: apply directly.
            """
            Tell Pylance exactly what to expect using cast():
                - Clarifies the return type explicitly to the type checker.
                - Prevents incorrect tuple assumptions.
                - Keeps the code readable and safe.
            """
            filtered = cast(NDArray[np.float64], sosfilt(sos, data))
            return filtered
        elif data.ndim == 2:
            # Multi-channel EEG: apply to each channel independently.
            return np.array([sosfilt(sos, ch) for ch in data])
        else:
            raise ValueError('Input data must be 1D or 2D NumPy array.')

    """
    Z-score Normalisation:
        - This method standarises the EEG signal: mean = 0, std = 1.
        - Helps many ML algorithms work better and makes signal from different
          channels comparable.
    By the end of the method:
        - Classic z-score formula: substract mean and divide by standard deviation.
        - Works for both 1D and 2D arrays (NumPy applies it element-wise).
    """
    def z_score_normalise(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalisation to EEG data.

        Args:
            data (np.ndarray): EEG signal (raw, filtered, or simulated).

        Returns:
            np.ndarray: Signal scaled to zero mean and unit variance.
        """
        return (data - np.mean(data)) / np.std(data)
