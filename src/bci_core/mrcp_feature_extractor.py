# src/bci_core/mrcp_feature_extractor.py

'''
MRCPs are slow negative EEG potentials observable in the 1-2 seconds before 
voluntary movement.

They are event-related potentials (ERP) and reflect brain activity in the 
supplementary motor area, premotor cortex, and primary motor cortex.

MRCP features are:
    - More robust to noise than pure motor imagery
    - Subject-independent to a degree (common morphology)
    - Time-domain based, enabling use with low-frequency hardware
    - Found in all healthy adults and most stroke patients (even if weak)

For each channel (especially Cz, C3, C4), extract:
    Feature	Method:
    - Area	Trapezoidal integration from -1.5 to 0 s
    - Negative peak	Minimum voltage in the window
    - Slope	Linear fit (least squares) from -1.0 to -0.5 s
    - Mahalanobis distance	From mean rest template
'''

import numpy
from typing import Dict

class MRCPFeatureExtractor: # Create the base class MRCPFeatureExtractor
    '''
    Motor-Related Cortical Potential (MRCP) Feature Extraction.

    This module computes key time-domain features from EEG trials
    aligned to voluntary motor intention, based on the morphology of
    MRCPs. These slow cortical potentials (SCPs) are essential biomarkers
    in non-invasive brain-machine interfaces, particularly in pre-movement
    detection for motor rehabilitation.

    Features extracted:
        - Area under the curve (integrated negativity)
        - Negative peak amplitude
        - Slope of the negative deflection
        - (Optional) Mahalanobis distance to baseline template

    Input Assumptions:
        - EEG trials are already filtered (e.g., 0.1-1.0 Hz bandpass)
        - Trials are baseline corrected (zero-mean at rest)
        - Shape: (n_channels, n_samples)
        - Sampling frequency: typically 250 Hz.
        - Epochs are time-locked to movement onset at t=0

    Attributes:
        sfreq (float): Sampling frequency in Hz (default: 250.0)
        t (np.ndarray): Time axis in seconds (computed per trial)
        results (Dict): Stores features per channel per trial
    '''

    def __init__(self, sfreq: float = 250.0): # Accept sampling frequency (sfreq).
        '''initialise the extractor with sampling rate

        Args:
            sfreq (float): Sampling frequency in Hz (default: 250.0)

        '''
        self.sfreq = sfreq # e.g. 250 Hz
        self.results = {}  # Prepare internal storage for features (self.results) 
                           # Populated after feature extraction
        self.t = None      # Time axis (set per trial)
                           # Prepares for future time vector (self.t) to allow 
                           # slicing windows in seconds, not just samples

    def extract_area(self, trial: np.ndarray) -> Dict[int, float]:
        '''
        Compute the area under the curve (AUC) of MRCP per channel.

        This method integrates the EEG signal from -1.5 to 0.0 seconds
        using the trapezoidal rule. It assumes the signal is already filtered
        and baseline corrected. The area is usually negative, reflecting the
        downward deflection of the MRCP waveform.

        Concept:
        - Instead of trying to find the exact area under a curve using calculus, the 
          trapezoidal rule approximates it by dividing the area into a series of 
          trapezoids. 
        - Each trapezoid's area is calculated using the formula: 
          (1/2) * height * (base1 + base2). 
        - The sum of these trapezoid areas provides an approximation of the total area 
          under the curve.

        -4.0 -> 0.0: is a pre-movement analysis window commonly used in MRCP studies.
        It includes:
          - The late Bereitschaftspotential (readiness potential).
          - The negative peak close to movement onset.
          - This window often contains the strongest motor-related signals.

        Args:
            trial (np.ndarray): EEG trial of shape (n_channels, n_samples).
                                The last sample corresponds to time t = 0.0.

        Returns:
            Dict[int, float]: Dictionary mapping channel index to AUC value.
        '''
        if trial.ndim != 2:
            raise ValueError('Input trial must be a 2D array (n_channels x n_samples)')

        n_channels, n_samples = trial.shape

        # Generate time axis in seconds. Assumes t = 0 is at the last column.
        self.t = np.linspace(-n_samples / self.sfreq, 0, n_samples) # 4, so (-4, 0, 1000).
                                                                    # 0 is the movement onset, and 
                                                                    # -4.0 is the start of the trial, to check 
                                                                    # MRCP (Motor-Related Cortical Potential).
        # Define time window for AUC computation: from -1.5 to 0.0 s.
        idx_window = np.where((self.t >= -1.5) & (self.t <= 0.0))[0]

        # Initialise dictionary to store AUC per channel.
        auc_per_channel = {}
        
        for ch in range(n_channels):
            # Extract signal segment within the defined window.
            segment = trial[ch, idx_window]

            # Extract corresponding time points.
            t_segment = self.t[idx_window]

            # Compute area using trapezoidal integration.
            auc = np.trapezoid(segment, x=t_segment)

            # Store in results.
            auc_per_channel[ch] = auc

    return auc_per_channel


