# src/bci_core/mrcp_feature_extractor.py

'''
Motor-Related Cortical Potential (MRCPs) are slow negative EEG potentials observable in the 1-2 seconds before 
voluntary movement. Most clearly seen over central electrodes (Cz, C3, C4).

They are event-related potentials (ERP) and reflect brain activity in the 
supplementary motor area, premotor cortex, and primary motor cortex.

MRCP features are:
    - More robust to noise than pure motor imagery
    - Subject-independent to a degree (common morphology)
    - Time-domain based, enabling use with low-frequency hardware
    - Found in all healthy adults and most stroke patients (even if weak)

For each channel (especially Cz, C3, C4), extract:
    Feature	Method:
    - Area	Trapezoidal integration from -1.5 to 0.0 s
    - Negative peak	Minimum voltage in the window
    - Slope	Linear fit (least squares) from -1.0 to -0.5 s
        - Least Squares Method is used to derive a generalized linear equation 
          between two variables. When the value of the dependent and independent 
          variables they are represented as x and y coordinates in a 2D Cartesian 
          coordinate system. Initially, known values are marked on a plot. The plot obtained at this point is called a scatter plot.
    - Mahalanobis distance
        - The Mahalanobis distance is the distance to the centre of a class taking 
          correlation into account and is the same for all points on the same 
          probability ellipse. For equally probable classes, i.e. classes with the 
          same number of training objects, a smaller Mahalanobis distance to class 
          K than to class L, means that the probability that the object belongs to 
          class K is larger than that it belongs to L.
'''

import numpy as np
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
            auc = -np.trapz(segment, x=t_segment)

            # Store in results.
            auc_per_channel[ch] = auc

        return auc_per_channel

    def extract_peak(self, trial):
        '''
        Compute the negative peak amplitude of Motor-Related Cortical Potential 
        (MRCP) per channel.
        
        This method scans each EEG channel to identify the most negative
        voltage value in the interval from -1.5 to 0.0 seconds. This peak
        typically occurs just before the voluntary movement and serves as
        a critical feature in MRCP-based classification tasks.

        Args:
            trial (np.ndarray): EEG trial of shape (n_channels, n_samples).
                                The last sample must correspond to time t = 0.0.

        Returns:
            Dict[int, float]: Dictionary mapping channel index to negative peak value.
        '''
        if trial.ndim != 2:
            raise ValueError ('Input trial must be a 2D array (n_channels x n_samples).')
        
        n_channels, n_samples = trial.shape

        if self.t is None or len(self.t) != n_samples:
            self.t = np.linspace(-n_samples / self.sfreq, 0, n_samples)

        # Define the time window of interest: [-1.5, 0.0] seconds.
        idx_window = np.where((self.t >= -1.5) & (self.t <= 0.0))[0]

        # Initialise dictionary to store peak values.
        peak_per_channel = {}

        for ch in range(n_channels):
            # Extract segment for this channel.
            segment = trial[ch, idx_window]

            # Compute the minimum (most negative) value.
            negative_peak = np.min(segment)

            # Store result.
            peak_per_channel[ch] = negative_peak
        
        return peak_per_channel

    def extract_slope(self, trial: np.ndarray) -> Dict[int, float]:
        '''
        Compute the slope of the MRCP deflection per channel:
        MRCP deflection refers to the downward (negative) shift in EEG amplitude 
        that occurs in the seconds leading up to a voluntary movement.
        A steeper slope means stronger or faster preparation, while a flatter one 
        might suggest delayed or reduced cortical involvement.
        This function extracts that slope per channel using linear regression in 
        the window prior to movement onset (t = 0), aiding in movement-related 
        signal characterisation.

        This method fits a straight line to the EEG signal between
        -1.0 and -0.5 seconds before movement onset using the least
        squares method. The slope quantifies the rate of the negative
        build-up and is often used to assess movement preparation dynamics.

        Args:
            trial (np.ndarray): EEG trial of shape (n_channels, n_samples).
                                The last sample must correspond to time t = 0.0.

        Returns:
            Dict[int, float]: Dictionary mapping channel index to slope value.
        '''
        if trial.ndim != 2:
            raise ValueError ('Input trial must be a 2D array (n_channel x samples).')
        
        n_channel, n_samples = trial.shape

        if self.t is None or len(self.t) != n_samples:
            # self.t generates a vector of n_samples values.
            self.t = np.linspace(-n_samples / self.sfreq, 0, n_samples) # Time at sample 0 = -4.0 seconds.
                                                                        # Time at sample 999 = 0.0 seconds.
                                                                        # Step = 4.0 / (1000 - 1) ≈ 0.004 
                                                                        # seconds per sample.
        
        # Define slope window: from -1.0 to -0.5.
        idx_window = np.where((self.t >= -1.0) & (self.t <= -0.5))[0]

        # Initialise output directly.
        slope_per_channel = {}

        for ch in range (n_channel):
            # Extract time an signal for slope computation.
            t_segment = self.t[idx_window]
            segment = trial[ch, idx_window]

            # Fit a first-degree polynomial (linear fit): y = mx + b
            coeffs = np.polyfit(t_segment, segment, deg=1) # Uses np.polyfit() to fit a line 
                                                           # y = mt + b over [−1.0, −0.5] seconds.

            # Extract the slope (coefficient of t).
            slope = coeffs[0]

            slope_per_channel[ch] = slope
        
        return slope_per_channel

    def extract_all(self, trial: np.ndarray) -> Dict[int, Dict[str, float]]:
        '''
        Extract all MRCP features for a given EEG trial.

        This method computes the area under the curve, negative peak,
        and slope for each channel and aggregates them into a single
        dictionary. Results are also stored internally for reuse.

        Args:
            trial (np.ndarray): EEG trial of shape (n_channels, n_samples).
                                The last sample must correspond to time t = 0.0.

        Returns:
            Dict[int, Dict[str, float]]: Nested dictionary of the form:
                results[channel] = {
                    'area': float,
                    'peak': float,
                    'slope': float
                }
        '''
        # Run each feature extractor separately.
        areas = self.extract_area(trial)
        peaks = self.extract_peak(trial)
        slope = self.extract_slope(trial)

        # Initialise final result dictionary.
        all_features = {}

        for ch in areas.keys():
            # Merge features per channel.
            all_features[ch] = {
                'area': areas[ch],
                'peak': peaks[ch],
                'slope': slope[ch]
            }

        # Store internally for downstream analysis or debugging.
        self.results = all_features

        return all_features
