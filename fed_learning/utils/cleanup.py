"""
Cleanup utility - Clean up temporary folders created during training.

Extracted from train_incremental_kaggle.py to be reusable.
"""

import os
import shutil
from typing import List, Optional


# Default temp folders created by various algorithms
DEFAULT_TEMP_FOLDERS = [
    "./temp_svd_storage",
    "./temp_ewc_storage",
    "./temp_fedlwf_storage",
    "./temp_test_data",
]


def cleanup_temp_folders(folders: Optional[List[str]] = None):
    """
    Clean up temporary folders.

    Args:
        folders: List of folder paths to clean. If None, uses default list.
    """
    if folders is None:
        folders = DEFAULT_TEMP_FOLDERS

    for folder in folders:
        if os.path.exists(folder):
            print(f"🧹 Cleaning {folder}...")
            shutil.rmtree(folder)
