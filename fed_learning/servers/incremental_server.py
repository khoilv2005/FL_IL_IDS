"""
Incremental Server - Specialized server for Class Incremental Learning.
Extends FederatedServer with optimized evaluation (skip AUC by default).
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
    Server optimized for Class Incremental Learning.
    
    Key differences from FederatedServer:
    - evaluate_global() skips AUC by default (faster, avoids warnings)
    - AUC only computed when explicitly requested (final task)
    """
    
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


# =============================================================================
# FedCBDR Server - Server with Global-perspective Data Replay (GDR) for FCIL
# =============================================================================
from collections import OrderedDict
from threading import Thread


class FedCBDRServer:
    """
    Server for FedCBDR algorithm with Global-perspective Data Replay.
    
    Reference:
        "Class-wise Balancing Data Replay for Federated Class-Incremental Learning"
        Zhuang Qi et al., arXiv:2507.07712, 2025
    
    Key components:
    1. Standard FL training with FedAvg aggregation
    2. GDR module for coordinated replay buffer updates
    3. Task-aware evaluation across all seen classes
    
    Attributes:
        clients: List of FedCBDRClient instances
        global_model: Global CNN-GRU model
        trainer: FedCBDRTrainer instance
        aggregator: FedCBDRAggregator instance
        leverage_calculator: For global leverage score computation
    """
    
    def __init__(
        self,
        clients,
        test_data: Dict,
        config: Dict
    ):
        """
        Initialize FedCBDR server.
        
        Args:
            clients: List of FedCBDR clients
            test_data: Test data dict with X_test, y_test
            config: Configuration dict
        """
        from ..models.cnn_gru import CNN_GRU_Model
        from ..strategies.incremental.fedcbdr import (
            FedCBDRTrainer, FedCBDRAggregator, LeverageScoreCalculator
        )
        
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.num_classes = config["num_classes"]
        
        # Device setup
        self.num_gpus = config.get("num_gpus") or torch.cuda.device_count()
        if self.num_gpus == 0:
            self.num_gpus = 1
            self.primary_device = "cpu"
            self.use_cpu = True
        else:
            self.primary_device = "cuda:0"
            self.use_cpu = False
        
        device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
        print(f"\n🖥️  FedCBDR Server: {device_info}, primary: {self.primary_device}")
        
        # Global model
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)
        
        # FedCBDR strategy
        self.trainer = FedCBDRTrainer(
            tau_old=config.get("tau_old", 0.9),
            tau_new=config.get("tau_new", 1.1),
            omega_old=config.get("omega_old", 1.1),
            omega_new=config.get("omega_new", 0.9),
        )
        self.aggregator = FedCBDRAggregator()
        
        # GDR components
        self.leverage_calculator = LeverageScoreCalculator(
            rank=config.get("leverage_rank", 50)
        )
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes = []
        self.task_classes: Dict[int, list] = {}
        
        # History
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1_macro": [],
            "test_f1_weighted": [],
            "test_precision_macro": [],
            "test_recall_macro": [],
            "test_auc_macro": [],
            "task_accuracies": [],
            "average_forgetting": [],
        }
        
        print(f"📊 Strategy: FedCBDR (Replay + TTS)")
        print(f"   τ_old={self.trainer.tau_old}, τ_new={self.trainer.tau_new}")
        print(f"   ω_old={self.trainer.omega_old}, ω_new={self.trainer.omega_new}")
    
    def get_global_params(self) -> OrderedDict:
        """Get global model parameters (CPU)."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )
    
    def set_global_params(self, params: OrderedDict):
        """Set global model parameters."""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )
    
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
        
        # Update trainer
        self.trainer.set_task(task_id, task_classes)
        
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
            compute_auc: Whether to compute AUC
            seen_classes_only: Only evaluate on seen classes
            
        Returns:
            Dict with metrics
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
            return {"accuracy": 0.0, "f1_macro": 0.0}
        
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
        
        # AUC
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
    
    def evaluate_per_task(self, batch_size: int = 1024) -> Dict[int, float]:
        """
        Evaluate accuracy per task (for forgetting analysis).
        
        Returns:
            Dict mapping task_id to accuracy on that task's classes
        """
        self.global_model.eval()
        
        task_accuracies = {}
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        
        for task_id, task_classes in self.task_classes.items():
            if not task_classes:
                continue
            
            # Filter test data for this task's classes
            task_class_set = set(task_classes)
            mask = torch.tensor([y.item() in task_class_set for y in y_test])
            
            if not mask.any():
                task_accuracies[task_id] = 0.0
                continue
            
            X_task = X_test[mask]
            y_task = y_test[mask]
            
            # Evaluate
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for i in range(0, len(y_task), batch_size):
                    X_batch = X_task[i:i+batch_size].to(self.primary_device)
                    y_batch = y_task[i:i+batch_size]
                    
                    out = self.global_model(X_batch)
                    preds = out.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.numpy())
            
            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)
        
        return task_accuracies
    
    def compute_average_forgetting(self) -> float:
        """
        Compute Average Forgetting (AF) metric.
        
        AF = (1/T-1) * Σ_{t=0}^{T-2} max_{t'≤T-1} (a_{t',t} - a_{T-1,t})
        
        Returns:
            Average forgetting value
        """
        if self.current_task == 0:
            return 0.0
        
        current_accs = self.evaluate_per_task()
        
        # Update trainer's tracking
        self.trainer.update_forgetting(current_accs)
        
        return self.trainer.last_af
    
    def save_checkpoint(self, path: str, task_id: int):
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "task_id": task_id,
            "model_state_dict": self.global_model.state_dict(),
            "seen_classes": self.seen_classes,
            "task_classes": self.task_classes,
            "config": self.config,
            "history": self.history,
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.primary_device)
        
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.seen_classes = checkpoint["seen_classes"]
        self.task_classes = checkpoint["task_classes"]
        self.current_task = checkpoint["task_id"]
        self.history = checkpoint.get("history", self.history)
        
        print(f"📂 Checkpoint loaded: {path}")
        print(f"   Task: {self.current_task}, Seen classes: {len(self.seen_classes)}")


# =============================================================================
# FedLwF Server - Server for Federated Learning without Forgetting
# =============================================================================

class FedLwFServer:
    """
    Server for FedLwF (Federated Learning without Forgetting).
    
    Implements FedAvg aggregation with knowledge distillation for
    class-incremental learning.
    
    Key Features:
    - Standard FedAvg model aggregation
    - Coordinates model snapshot distribution to clients
    - Task-aware evaluation and forgetting metrics
    - Multi-GPU support
    
    Attributes:
        clients: List of FedLwFClient instances
        global_model: Global CNN-GRU model
        trainer: FedLwFTrainer instance
        aggregator: FedLwFAggregator (FedAvg)
    """
    
    def __init__(
        self,
        clients,
        test_data: Dict,
        config: Dict
    ):
        """
        Initialize FedLwF server.
        
        Args:
            clients: List of FedLwF clients
            test_data: Test data dict with X_test, y_test
            config: Configuration dict
        """
        from ..models.cnn_gru import CNN_GRU_Model
        from ..strategies.incremental.fedlwf import FedLwFTrainer, FedLwFAggregator
        
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.num_classes = config["num_classes"]
        
        # Device setup
        self.num_gpus = config.get("num_gpus") or torch.cuda.device_count()
        if self.num_gpus == 0:
            self.num_gpus = 1
            self.primary_device = "cpu"
            self.use_cpu = True
        else:
            self.primary_device = "cuda:0"
            self.use_cpu = False
        
        device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
        print(f"\n🖥️  FedLwF Server: {device_info}, primary: {self.primary_device}")
        
        # Global model
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)
        
        # FedLwF strategy
        self.trainer = FedLwFTrainer(
            lwf_alpha=config.get("lwf_alpha", 1.0),
            temperature=config.get("temperature", 2.0),
            distill_on_new_only=config.get("distill_on_new_only", False),
            temp_dir=config.get("temp_dir", "./temp_fedlwf_storage"),
        )
        self.aggregator = FedLwFAggregator()
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes = []
        self.task_classes: Dict[int, list] = {}
        
        # History
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1_macro": [],
            "test_f1_weighted": [],
            "test_precision_macro": [],
            "test_recall_macro": [],
            "test_auc_macro": [],
            "task_accuracies": [],
            "average_forgetting": [],
        }
        
        print(f"📊 Strategy: FedLwF (FedAvg + Knowledge Distillation)")
        print(f"   α={self.trainer.lwf_alpha}, T={self.trainer.temperature}")
    
    def get_global_params(self) -> OrderedDict:
        """Get global model parameters (CPU)."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )
    
    def set_global_params(self, params: OrderedDict):
        """Set global model parameters."""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )
    
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
        
        # Update trainer
        self.trainer.set_task(task_id, task_classes)
        
        print(f"\n📌 Task {task_id}: classes {task_classes}")
        print(f"   Total seen classes: {len(self.seen_classes)}")
    
    def save_global_snapshot(self):
        """
        Save global model snapshot after task completion.
        
        This snapshot is distributed to clients for KD in next task.
        """
        # Save in trainer (will be distributed via config)
        self.trainer.save_model_snapshot(self.global_model)
        
        # Also save snapshot to all clients
        global_state = self.get_global_params()
        for client in self.clients:
            client.old_model_state = OrderedDict(
                (k, v.clone()) for k, v in global_state.items()
            )
            client.old_model = None  # Clear cache, will reload
    
    def train_round(
        self,
        participating_clients=None,
        verbose: bool = True
    ) -> Dict:
        """
        Train one federated round with FedAvg.
        
        Args:
            participating_clients: Clients to train (default: all)
            verbose: Whether to print progress
            
        Returns:
            Dict with train_loss and round_time
        """
        import time
        from ..training.fedlwf_worker import train_fedlwf_clients_on_gpu
        
        round_start = time.time()
        
        clients = participating_clients or self.clients
        
        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n→ FedLwF: Training {len(clients)} clients on {device_info}")
        
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
                    target=train_fedlwf_clients_on_gpu,
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
        
        return {"train_loss": avg_loss, "round_time": round_time}
    
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
            compute_auc: Whether to compute AUC
            seen_classes_only: Only evaluate on seen classes
            
        Returns:
            Dict with metrics
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
            return {"accuracy": 0.0, "f1_macro": 0.0}
        
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
        
        # AUC
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
    
    def evaluate_per_task(self, batch_size: int = 1024) -> Dict[int, float]:
        """
        Evaluate accuracy per task (for forgetting analysis).
        
        Returns:
            Dict mapping task_id to accuracy on that task's classes
        """
        self.global_model.eval()
        
        task_accuracies = {}
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        
        for task_id, task_classes in self.task_classes.items():
            if not task_classes:
                continue
            
            # Filter test data for this task's classes
            task_class_set = set(task_classes)
            mask = torch.tensor([y.item() in task_class_set for y in y_test])
            
            if not mask.any():
                task_accuracies[task_id] = 0.0
                continue
            
            X_task = X_test[mask]
            y_task = y_test[mask]
            
            # Evaluate
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for i in range(0, len(y_task), batch_size):
                    X_batch = X_task[i:i+batch_size].to(self.primary_device)
                    y_batch = y_task[i:i+batch_size]
                    
                    out = self.global_model(X_batch)
                    preds = out.argmax(dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.numpy())
            
            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)
        
        return task_accuracies
    
    def compute_average_forgetting(self) -> float:
        """
        Compute Average Forgetting (AF) metric.
        
        Returns:
            Average forgetting value
        """
        if self.current_task == 0:
            return 0.0
        
        current_accs = self.evaluate_per_task()
        
        # Update trainer's tracking
        self.trainer.update_forgetting(current_accs)
        
        return self.trainer.last_af
    
    def save_checkpoint(self, path: str, task_id: int):
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "task_id": task_id,
            "model_state_dict": self.global_model.state_dict(),
            "seen_classes": self.seen_classes,
            "task_classes": self.task_classes,
            "config": self.config,
            "history": self.history,
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.primary_device)
        
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.seen_classes = checkpoint["seen_classes"]
        self.task_classes = checkpoint["task_classes"]
        self.current_task = checkpoint["task_id"]
        self.history = checkpoint.get("history", self.history)
        
        print(f"📂 Checkpoint loaded: {path}")
