"""
Data Preparation Pipeline
=========================

Scripts for preparing and preprocessing data for federated learning.

Pipeline Steps:
    1. step1_prepare_chunks.py   - Raw CSV → chunked NPZ files (label encoding only)
    2. step2_federated_splits.py - Chunks → federated client splits (train-only scaling)
    3. step3_visualize.py        - Visualize data distributions

Utility Scripts:
    - check_class_data.py     - Verify class distribution across clients
    - check_participation.py  - Verify client participation per task
    - detect_label.py         - Quick label detection utility
"""
