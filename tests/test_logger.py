# tests/test_logger.py
"""
Simulate a single EEG sample and log it.
"""

import os
import random
from bci_core.logger import BCILogger

# Method to test with only 1 EEG sample if eeg_logs.csv is created.
def test_logger_one_sample():
    """
    Test that BCILogger creates a CSV file and logs a single EEG sample correctly.
    """
    
    # Path to test log file.
    log_path = 'logs/eeg_log.csv'

    # Ensure a clean slate by removing any existing log file.
    if os.path.exists(log_path):
        os.remove(log_path)

    # Create the logger instance (uses default: logs/eeg_log.csv).
    logger = BCILogger()

    # Generate 16 random float values to simulate one EEG timepoint.
    # EEG signals typically range ±100 µV, so this simulates real-world values.
    sample = [random.uniform(-100.0, 100.0) for _ in range(16)]
    logger.log(sample)

    # Check: file was created.
    assert os.path.exists(log_path), 'Log file was not created.'

    # Read the file to check its contents.
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Expect 2 lines: one header + one sample.
    assert len(lines) == 2, f'Expected 2 lines (header + date), found {len(lines)}.'

    # Optional: verify that header line contains 17 columns (timestamp + ch1 to ch16).
    # assert <condition>, <error_message>. If <condition> is False, Python raises:
    # AssertionError: <error_message>.
    header = lines[0].strip().split(',')
    assert header[0] == 'timestamp', "'Header missing timestamp'"
    assert len(header) == 17, f'Expected 17 columns in header, found {len(header)}'

# Method to test if whith 5 EEG samples eeg_logs.csv is created.
def test_logger_mult_samples():
    """
    Test that BCILogger correctly appends multiple EEG samples to the same file.
    """
    log_path = 'logs/eeg_log.csv'

    # Clean up.
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = BCILogger()

    # Log 5 EEG samples (each with 16 channels).
    for _ in range(5):
        sample = [random.uniform(-100.0, 100.0) for _ in range(16)]
        logger.log(sample)

    # Verify the file has 1 header + 5 data lines.
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 6, f'Expected 6 lines (header + 5 samples), got {len(lines)}'