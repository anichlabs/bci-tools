# Anich Labs: BCI Tools

This repository contains the core modules and research pipeline for Brain–Computer Interface (BCI) development at Anich Labs. Based on the NeuroExo architecture, this implementation is adapted for solo R&D using OpenBCI equipment and modular Python components hosted on fully GDPR-compliant European infrastructure.

## Directory Structure

bci-tools/
├── data/
│ ├── raw/ # Unprocessed EEG signals (.csv, .tdms)
│ ├── clean/ # Filtered, artefact-free segments
│ ├── feat/ # Movement-related cortical potential (MRCP) features
│ └── classified/ # SVM-based movement intent classifications
└── ...

## Core Modules

- `logger/`: EEG acquisition and file storage
- `signal_preprocessor/`: Filtering, artefact rejection, trial segmentation
- `feature_extractor/`: MRCP features including area, slope, peak, Mahalanobis distance
- `intent_classifier/`: Support Vector Machine model for detecting movement intent
- `ui/`: User interface (CLI initially; Tkinter or React in future phases)

## Hardware and Infrastructure

- EEG hardware: OpenBCI All-in-One Biosensing R&D Bundle
- Local development: OpenSUSE (ZSH), Python 3.10, Conda, CUDA
- Remote infrastructure: Hetzner VPS CX22, GitLab CI/CD, European backups

## Compliance

- Raw data encrypted prior to transmission or storage
- Entire infrastructure is hosted within the European Union (GDPR-aligned)
- Roadmap and architecture follow Model Context Protocol (MCP) and Modular Clinical Pipeline guidelines

## Stage 1 Objectives

- Establish data directory structure
- Finalise core signal preprocessing and feature extraction modules
- Train basic movement intent classifier (SVM-based)
- Sync pipeline to Hetzner VPS via Git
- Begin development of the user interface
