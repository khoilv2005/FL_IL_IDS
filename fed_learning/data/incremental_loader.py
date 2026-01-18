"""
Incremental Data Loader - Load data by task sequence for Class Incremental Learning.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

import torch
import numpy as np


class IncrementalDataLoader:
    """
    Load data in task sequence for Class Incremental Learning.
    
    Example with 34 classes and 8 tasks:
        Task 1: 5 classes (Benign + 4 attacks)
        Task 2-8: +4 classes each
    """
    
    def __init__(
        self,
        data_dir: str,
        num_clients: int = 100,
        base_classes: int = 5,
        classes_per_task: int = 4,
        total_classes: int = 34,
    ):
        self.data_dir = Path(data_dir)
        self.num_clients = num_clients
        self.base_classes = base_classes
        self.classes_per_task = classes_per_task
        self.total_classes = total_classes
        
        # Calculate number of tasks
        remaining = total_classes - base_classes
        self.num_tasks = 1 + (remaining + classes_per_task - 1) // classes_per_task
        
        # Generate task -> classes mapping
        self.task_classes = self._generate_task_classes()
        
        # Cache loaded data
        self._raw_data: Optional[Dict] = None
        self._test_data: Optional[Dict] = None
        self.input_shape = None
    
    def _generate_task_classes(self) -> Dict[int, List[int]]:
        """Generate mapping from task_id to list of classes."""
        task_classes = {}
        
        # Task 0: base classes
        task_classes[0] = list(range(self.base_classes))
        
        # Subsequent tasks
        current_class = self.base_classes
        for task_id in range(1, self.num_tasks):
            end_class = min(current_class + self.classes_per_task, self.total_classes)
            task_classes[task_id] = list(range(current_class, end_class))
            current_class = end_class
        
        return task_classes
    
    def get_classes_for_task(self, task_id: int) -> List[int]:
        """Get new classes introduced in a specific task."""
        return self.task_classes.get(task_id, [])
    
    def get_seen_classes(self, task_id: int) -> List[int]:
        """Get all classes seen up to and including task_id."""
        seen = []
        for t in range(task_id + 1):
            seen.extend(self.task_classes.get(t, []))
        return seen
    
    def load_raw_data(self):
        """Load all raw data once (called internally)."""
        if self._raw_data is not None:
            return
        
        print(f"\n📥 Loading raw data from {self.data_dir}...")
        
        self._raw_data = {}
        
        for cid in range(self.num_clients):
            # Use .npz format (consistent with loader.py)
            path = self.data_dir / f"client_{cid}_train.npz"
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            
            data = np.load(path)
            X = torch.from_numpy(data['X_train'].astype(np.float32))
            y = torch.from_numpy(data['y_train'].astype(np.int64))
            
            self._raw_data[cid] = {"X": X, "y": y}
            
            if self.input_shape is None:
                self.input_shape = tuple(X.shape[1:])
        
        # Load test data
        test_path = self.data_dir / "global_test_data.npz"
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing: {test_path}")
        
        test_npz = np.load(test_path)
        self._test_data = {
            "X_test": torch.from_numpy(test_npz['X_test'].astype(np.float32)),
            "y_test": torch.from_numpy(test_npz['y_test'].astype(np.int64)),
        }
        
        print(f"✓ Loaded {self.num_clients} clients, input_shape: {self.input_shape}")
    
    def get_task_data(
        self, 
        task_id: int
    ) -> Tuple[Dict[int, Dict], Dict, List[int]]:
        """
        Get data filtered for a specific task.
        
        Returns:
            (client_data, test_data, new_classes)
        """
        self.load_raw_data()
        
        new_classes = self.get_classes_for_task(task_id)
        seen_classes = self.get_seen_classes(task_id)
        
        print(f"\n📋 Task {task_id}: new classes = {new_classes}, seen = {len(seen_classes)}")
        
        # Filter client data - only new classes for training
        client_data = {}
        for cid in range(self.num_clients):
            X = self._raw_data[cid]["X"]
            y = self._raw_data[cid]["y"]
            
            # Filter to new classes only
            mask = torch.zeros(len(y), dtype=torch.bool)
            for c in new_classes:
                mask = mask | (y == c)
            
            if mask.sum() > 0:
                client_data[cid] = {
                    "X_train": X[mask],
                    "y_train": y[mask],
                }
            else:
                client_data[cid] = {
                    "X_train": torch.empty(0, *self.input_shape),
                    "y_train": torch.empty(0, dtype=torch.long),
                }
        
        # Filter test data - all seen classes
        X_test = self._test_data["X_test"]
        y_test = self._test_data["y_test"]
        
        test_mask = torch.zeros(len(y_test), dtype=torch.bool)
        for c in seen_classes:
            test_mask = test_mask | (y_test == c)
        
        test_data = {
            "X_test": X_test[test_mask],
            "y_test": y_test[test_mask],
        }
        
        return client_data, test_data, new_classes
    
    def __repr__(self):
        return (
            f"IncrementalDataLoader(tasks={self.num_tasks}, "
            f"classes={self.total_classes}, "
            f"base={self.base_classes}, per_task={self.classes_per_task})"
        )
