"""
CGoFed Server - Standard Incremental Server for CGoFed and EWC.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from .server import FederatedServer


class IncrementalServer(FederatedServer):
    """
    Server optimized for Class Incremental Learning (CGoFed, EWC).
    
    Key differences from FederatedServer:
    - evaluate_global() skips AUC by default (faster).
    - AUC only computed when explicitly requested (final task).
    - Tracks seen classes for incremental evaluation.
    """
    
    def __init__(
        self,
        clients,
        test_data: Dict,
        config: Dict
    ):
        """Initialize IncrementalServer."""
        super().__init__(clients, test_data, config)
        
        self.seen_classes = []
        self.task_classes: Dict[int, list] = {}
        self.current_task = 0

    def set_task(self, task_id: int, task_classes: list):
        """
        Set up for a new task.
        
        Args:
            task_id: Task identifier
            task_classes: List of class IDs in this task
        """
        self.current_task = task_id
        self.task_classes[task_id] = task_classes
        self.seen_classes.extend(task_classes)
        
        print(f"\n📌 Task {task_id}: classes {task_classes}")
        print(f"   Total seen classes: {len(self.seen_classes)}")
    
    def evaluate_global(
        self, 
        batch_size: int = 1024, 
        compute_auc: bool = False,
        seen_classes_only: bool = True
    ) -> Dict:
        """
        Evaluate global model on test set.
        
        Args:
            batch_size: Batch size for evaluation
            compute_auc: Whether to compute AUC (slow, only needed at final task)
            seen_classes_only: Only evaluate on seen classes (for incremental learning)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        
        # Filter to seen classes if requested
        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]
            
        n_test = len(y_test)
        if n_test == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0
            }
        
        all_preds = []
        all_targets = []
        all_proba = [] if compute_auc else None
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i:i+batch_size].to(self.primary_device)
                y_batch = y_test[i:i+batch_size].to(self.primary_device)
                
                out = self.global_model(X_batch)
                loss = criterion(out, y_batch)
                total_loss += loss.item() * len(y_batch)
                
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
                if compute_auc:
                    proba = F.softmax(out, dim=1)
                    all_proba.append(proba.cpu().numpy())
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        
        metrics = {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # AUC - only compute if requested
        metrics["auc_macro_ovr"] = None
        if compute_auc and all_proba:
            try:
                y_proba = np.vstack(all_proba)
                y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
                if y_true_bin.shape[1] == 1:
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                metrics["auc_macro_ovr"] = roc_auc_score(
                    y_true_bin, y_proba, average="macro", multi_class="ovr"
                )
            except Exception:
                pass
        
        return metrics

# Alias for clarity
CGoFedServer = IncrementalServer
