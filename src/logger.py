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

# src/logger.py

import os

class BCILogger:
    """
    BCILogger is responsible for logging EEG data into a CSV file.
    It will be used to save time-stamped 16-channel EEG samples,
    either from real OpenBCI data or simulated signals.
    """

    def __init__(self, filename: str = 'logs/eeg_log.csv') -> None: # self refers to the current instance of the class. Think of it as the object in memory
                                                                    # -> None: __init__ must return None, and it cannot return any value.
        """
        Constructor: sets the filename, ensures directory exists,
        and creates the file with header if it doesn't already exist.

        Args:
            filename (str): Relative or absolute path to the CSV log file.
                            Defaults to 'logs/eeg_log.csv'.
        """
        # Store the filename as an instance attribute.
        self.filename = filename

        # Create the log directory if it doesn't exist.
        log_dir = os.path.dirname(self.filename)
        os.makedirs(log_dir, exist_ok=True)

        # Create the file if it doesn't exist and write a header.
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as f:
                # Write the CSV header: timestamp, ch1, ch2, ..., ch16.
                header = 'timestamp,' + ','.join([f'ch{i+1}' for i in range(16)]) + '\n'
                f.write(header)
