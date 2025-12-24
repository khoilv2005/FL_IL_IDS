"""
Federated Server with Multi-GPU Support
"""

import time
from collections import OrderedDict
from typing import Dict, List
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from ..models.cnn_gru import CNN_GRU_Model
from ..clients.client import FederatedClientMultiGPU
from ..training.gpu_trainer import train_clients_on_gpu


class FederatedServerMultiGPU:
    """
    Server hỗ trợ Multi-GPU training.
    """
    
    def __init__(self, clients: List[FederatedClientMultiGPU], 
                 test_data: Dict, config: Dict):
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.num_classes = config["num_classes"]
        
        # Detect GPUs
        self.num_gpus = config.get("num_gpus") or torch.cuda.device_count()
        if self.num_gpus == 0:
            self.num_gpus = 1  # Treat as 1 "device" for CPU
            self.primary_device = "cpu"
            self.use_cpu = True
        else:
            self.primary_device = "cuda:0"
            self.use_cpu = False
        
        device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
        print(f"\n🖥️  Detected {device_info}, primary device: {self.primary_device}")
        
        # Global model (trên primary device để eval)
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)
        
        # Velocity for FedAvgM
        self.velocity = OrderedDict(
            (k, torch.zeros_like(v))
            for k, v in self.global_model.state_dict().items()
        )
        
        # Server momentum params
        self.server_momentum = config.get("server_momentum", 0.9)
        self.server_lr = config.get("server_lr", 1.0)
        
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
        }
    
    def get_global_params(self) -> OrderedDict:
        """Lấy params của global model (CPU)"""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )
    
    def set_global_params(self, params: OrderedDict):
        """Set params cho global model"""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )
    
    def aggregate_fedavg(self, results: List[Dict]) -> OrderedDict:
        """Weighted average aggregation"""
        total_samples = sum(r["num_samples"] for r in results)
        
        agg = None
        for r in results:
            w_i = r["num_samples"] / max(1, total_samples)
            params = r["params"]
            
            if agg is None:
                agg = OrderedDict((k, w_i * v.float()) for k, v in params.items())
            else:
                for k in agg.keys():
                    if agg[k].dtype.is_floating_point:
                        agg[k] = agg[k] + w_i * params[k].float()
                    else:
                        agg[k] = params[k]
        
        return agg
    
    def train_round(self, verbose: bool = True) -> Dict:
        """
        Train 1 round với Multi-GPU.
        Chia clients cho các GPUs và train song song.
        """
        algo = self.config["algorithm"].lower()
        round_start = time.time()
        
        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n→ {algo.upper()}: Training {len(self.clients)} clients on {device_info}")
        
        global_params = self.get_global_params()
        
        # Chia clients cho các GPUs (hoặc 1 group cho CPU)
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(self.clients):
            clients_per_gpu[i % self.num_gpus].append(c)
        
        if verbose:
            for gpu_id, clients in enumerate(clients_per_gpu):
                device_label = "CPU" if self.use_cpu else f"GPU {gpu_id}"
                print(f"   {device_label}: {len(clients)} clients")
        
        # Results dict shared giữa các threads
        results_dict = {}
        
        # Tạo threads - mỗi thread train trên 1 GPU (hoặc 1 thread cho CPU)
        threads = []
        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_clients_on_gpu,
                    args=(gpu_id, clients_per_gpu[gpu_id], global_params, 
                          self.config, results_dict, algo, self.use_cpu)
                )
                threads.append(t)
                t.start()
        
        # Đợi tất cả threads hoàn thành
        for t in threads:
            t.join()
        
        # Collect results theo thứ tự client_id
        results = [results_dict[i] for i in range(len(self.clients))]
        
        # Aggregate
        w_avg = self.aggregate_fedavg(results)
        
        # Apply FedAvgM momentum nếu cần
        if algo == "fedavgm":
            w_t = self.get_global_params()
            beta = self.server_momentum
            new_params = OrderedDict()
            
            for k in w_t.keys():
                if w_t[k].dtype.is_floating_point:
                    delta = w_avg[k] - w_t[k]
                    new_v = beta * self.velocity[k] + delta
                    self.velocity[k] = new_v
                    new_params[k] = w_t[k] + self.server_lr * new_v
                else:
                    new_params[k] = w_avg[k]
            
            self.set_global_params(new_params)
        else:
            self.set_global_params(w_avg)
        
        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start
        
        if verbose:
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")
        
        return {"train_loss": avg_loss, "round_time": round_time}
    
    def evaluate_global(self, batch_size: int = 1024) -> Dict:
        """Evaluate global model trên test set"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        n_test = len(y_test)
        
        all_preds = []
        all_targets = []
        all_proba = []
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i:i+batch_size].to(self.primary_device)
                y_batch = y_test[i:i+batch_size].to(self.primary_device)
                
                out = self.global_model(X_batch)
                loss = criterion(out, y_batch)
                total_loss += loss.item() * len(y_batch)
                
                proba = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_proba.append(proba.cpu().numpy())
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_proba = np.vstack(all_proba)
        
        metrics = {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # AUC
        try:
            y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            metrics["auc_macro_ovr"] = roc_auc_score(
                y_true_bin, y_proba, average="macro", multi_class="ovr"
            )
        except Exception as e:
            metrics["auc_macro_ovr"] = None
        
        return metrics
