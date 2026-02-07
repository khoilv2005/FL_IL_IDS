"""Data loading module"""

from .loader import load_all_client_data_to_ram
from .incremental_loader import IncrementalDataLoader

__all__ = ["load_all_client_data_to_ram", "IncrementalDataLoader"]
