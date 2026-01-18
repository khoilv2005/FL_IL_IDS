"""
Incremental Server - Specialized server for Class Incremental Learning.
Extends FederatedServer with optimized evaluation (skip AUC by default).
"""

from typing import Dict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from threading import Thread

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from .server import FederatedServer
from ..training.worker import train_clients_on_gpu


class IncrementalServer(FederatedServer):
    """
    Server optimized for Class Incremental Learning.
    
    Key differences from FederatedServer:
    - evaluate_global() skips AUC by default (faster, avoids warnings)
    - AUC only computed when explicitly requested (final task)
    - Supports multiple workers per GPU for faster training
    """
    
    def train_round(self, verbose: bool = True) -> Dict:
        """
        Train one round with Multi-Worker Multi-GPU support.
        
        Uses config['workers_per_gpu'] to run multiple training threads per GPU,
        significantly speeding up training when VRAM is underutilized.
        """
        round_start = time.time()
        
        # Get workers_per_gpu (default 1 for backwards compatibility)
        workers_per_gpu = self.config.get("workers_per_gpu", 1)
        total_workers = self.num_gpus * workers_per_gpu
        
        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s) x {workers_per_gpu} workers"
            print(f"\n→ {self.config['algorithm'].upper()}: Training {len(self.clients)} clients on {device_info}")
        
        global_params = self.get_global_params()
        
        # Distribute clients across workers (not just GPUs)
        clients_per_worker = [[] for _ in range(total_workers)]
        for i, c in enumerate(self.clients):
            clients_per_worker[i % total_workers].append(c)
        
        if verbose:
            for worker_id in range(total_workers):
                gpu_id = worker_id % self.num_gpus
                device_label = "CPU" if self.use_cpu else f"GPU {gpu_id} Worker {worker_id}"
                print(f"   {device_label}: {len(clients_per_worker[worker_id])} clients")
        
        # Shared results dict
        results_dict = {}
        
        # Create threads for each worker
        threads = []
        for worker_id in range(total_workers):
            gpu_id = worker_id % self.num_gpus  # Round-robin GPU assignment
            
            if len(clients_per_worker[worker_id]) > 0:
                t = Thread(
                    target=train_clients_on_gpu,
                    args=(
                        gpu_id, 
                        clients_per_worker[worker_id], 
                        global_params,
                        self.config, 
                        results_dict, 
                        self.trainer,
                        self.use_cpu
                    )
                )
                threads.append(t)
                t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Collect results
        results = list(results_dict.values())
        
        # Aggregate using strategy
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)
        
        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start
        
        if verbose:
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")
        
        return {"train_loss": avg_loss, "round_time": round_time}
    
    def evaluate_global(self, batch_size: int = 1024, compute_auc: bool = False) -> Dict:
        """
        Evaluate global model on test set.
        
        Args:
            batch_size: Batch size for evaluation
            compute_auc: Whether to compute AUC (slow, only needed at final task)
                         Default False to skip AUC and avoid warnings during incremental tasks.
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        n_test = len(y_test)
        
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
        
        # AUC - only compute if requested (slow and requires all classes)
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
