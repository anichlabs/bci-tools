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