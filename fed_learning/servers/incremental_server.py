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
    
    def train_round(
        self,
        participating_clients=None,
        verbose: bool = True
    ) -> Dict:
        """
        Train one federated round.
        
        Args:
            participating_clients: Clients to train (default: all)
            verbose: Whether to print progress
            
        Returns:
            Dict with train_loss and round_time
        """
        import time
        import gc
        from ..training.fedcbdr_worker import train_fedcbdr_clients_on_gpu
        
        round_start = time.time()
        
        clients = participating_clients or self.clients
        
        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n→ FedCBDR: Training {len(clients)} clients on {device_info}")
        
        global_params = self.get_global_params()
        
        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(clients):
            clients_per_gpu[i % self.num_gpus].append(c)
        
        if verbose:
            for gpu_id, gpu_clients in enumerate(clients_per_gpu):
                device_label = "CPU" if self.use_cpu else f"GPU {gpu_id}"
                print(f"   {device_label}: {len(gpu_clients)} clients")
        
        # Train clients in parallel
        results_dict = {}
        threads = []
        
        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_fedcbdr_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        self.config,
                        results_dict,
                        self.trainer,
                        self.use_cpu
                    )
                )
                threads.append(t)
                t.start()
        
        for t in threads:
            t.join()
        
        # Collect results
        results = list(results_dict.values())
        
        # Aggregate (FedAvg)
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)
        
        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start
        
        if verbose:
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")
            
            # Print replay stats
            total_replay = sum(r.get("replay_samples", 0) for r in results)
            if total_replay > 0:
                print(f"→ Total replay samples: {total_replay}")
        
        return {"train_loss": avg_loss, "round_time": round_time}
    
    def coordinate_gdr(
        self,
        participating_clients=None,
        verbose: bool = True
    ):
        """
        Coordinate Global-perspective Data Replay (GDR).
        
        This is called after task completion to update replay buffers
        using globally-coordinated leverage scores.
        
        Paper Section 4.1: GDR Module
        1. Clients extract and encrypt features
        2. Server aggregates and computes global SVD
        3. Server computes leverage scores
        4. Server sends selection indices back to clients
        5. Clients update their replay buffers
        
        MEMORY OPTIMIZED: Process one client at a time to avoid OOM on Kaggle.
        
        Args:
            participating_clients: Clients that participated in this task
            verbose: Whether to print progress
        """
        import gc
        
        clients = participating_clients or self.clients
        
        if verbose:
            print(f"\n🔄 GDR: Coordinating replay buffer updates for {len(clients)} clients")
        
        # Move model to CPU to free GPU memory for feature extraction
        device = next(self.global_model.parameters()).device
        self.global_model.eval()
        
        # Process clients ONE AT A TIME to save memory
        processed_count = 0
        
        for idx, client in enumerate(clients):
            if client.num_samples == 0:
                continue
            
            if verbose:
                print(f"   Processing client {client.client_id} ({idx+1}/{len(clients)})...")
            
            try:
                # Step 1: Extract features for this client only
                features = client.extract_features_for_gdr(
                    self.global_model,
                    batch_size=128  # Smaller batch to save memory
                )
                
                if features is None or len(features) == 0:
                    if verbose:
                        print(f"   ⚠️ No features for client {client.client_id}")
                    continue
                
                # Step 2: Compute leverage scores
                scores = self.leverage_calculator.compute_scores(features)
                
                # Step 3: Select top samples based on leverage scores
                buffer_per_class = client.buffer_size // max(1, len(self.seen_classes))
                n_select = min(
                    buffer_per_class * len(client.current_classes),
                    len(scores)
                )
                
                _, top_indices = scores.topk(min(n_select, len(scores)))
                
                # Step 4: Update client's replay buffer
                client.update_replay_buffer(
                    self.global_model,
                    selected_indices=top_indices,
                    use_herding=False  # Disable herding to save memory
                )
                
                processed_count += 1
                
                # CRITICAL: Clean up immediately after each client
                del features, scores, top_indices
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ❌ Error processing client {client.client_id}: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        if verbose:
            print(f"   ✅ GDR complete: {processed_count} clients updated")
        
        if verbose:
            total_buffer = sum(c.replay_buffer.total_samples for c in clients)
            print(f"   GDR complete. Total buffer samples: {total_buffer}")
    
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
