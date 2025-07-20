# tests/test_logger.py
"""
Simulate a single EEG sample and log it.
"""

import os
import random
from src.logger import BCILogger

def test_logger():
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
    header = lines[0].strip().split(',')
    assert header[0] == 'timestamp', "'Header missing timestamp'"
    assert len(header) == 17, f'Expected 17 columns in header, found {len(header)}'