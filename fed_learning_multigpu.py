"""
Federated Learning with Multi-GPU Support
==========================================
- Data được load vào RAM (CPU)
- Training song song trên nhiều GPUs
- Mỗi GPU train một nhóm clients tuần tự
"""

import os
import json
import time
from datetime import datetime
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from threading import Thread
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "data_dir": "/kaggle/input/data-500clients",
    "output_dir": "./results",
    "checkpoint_dir": "./checkpoints",
    
    "num_clients": 500,
    "algorithm": "fedplus",  # fedavg, fedprox, fedavgm, fedplus
    
    # FedAvgM
    "server_momentum": 0.9,
    "server_lr": 1.0,
    
    # FedProx / Fed+
    "mu": 0.01,
    
    # Model (auto-detect)
    "input_shape": None,
    "num_classes": None,
    
    # Training
    "num_rounds": 5,
    "local_epochs": 3,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    
    # Multi-GPU
    "num_gpus": None,  # None = auto-detect
    
    # Eval & Save
    "eval_every": 1,
    "save_checkpoint_every": 1,
}


# =============================================================================
# MODEL: CNN-GRU
# =============================================================================
class CNN_GRU_Model(nn.Module):
    def __init__(self, input_shape, num_classes: int = 34):
        super().__init__()
        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
        else:
            seq_length = int(input_shape)

        self.input_shape = input_shape
        self.num_classes = num_classes

        # CNN blocks
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout_cnn1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout_cnn2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout_cnn3 = nn.Dropout(0.3)

        # Calculate CNN output size
        cnn_len = seq_length
        for _ in range(3):
            cnn_len = cnn_len // 2
        self.cnn_output_size = 256 * cnn_len

        # GRU
        self.gru1 = nn.GRU(1, 128, batch_first=True)
        self.gru2 = nn.GRU(128, 64, batch_first=True)
        self.gru_output_size = 64

        # MLP
        concat_size = self.cnn_output_size + self.gru_output_size
        self.dense1 = nn.Linear(concat_size, 256)
        self.bn_mlp1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.dense2 = nn.Linear(256, 128)
        self.bn_mlp2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)

        # CNN
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.dropout_cnn1(x_cnn)
        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.dropout_cnn2(x_cnn)
        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = self.dropout_cnn3(x_cnn)
        cnn_output = x_cnn.view(batch_size, -1)

        # GRU
        x_gru = x
        x_gru, _ = self.gru1(x_gru)
        x_gru, _ = self.gru2(x_gru)
        gru_output = x_gru[:, -1, :]

        # Concat & MLP
        z = torch.cat([cnn_output, gru_output], dim=1)
        z = self.dense1(z)
        if z.size(0) > 1:
            z = self.bn_mlp1(z)
        z = self.relu(z)
        z = self.dropout1(z)
        z = self.dense2(z)
        if z.size(0) > 1:
            z = self.bn_mlp2(z)
        z = self.relu(z)
        z = self.dropout2(z)
        return self.output(z)


# =============================================================================
# DATA LOADING - VÀO RAM (CPU), KHÔNG PHẢI GPU
# =============================================================================
def load_all_client_data_to_ram(data_dir: str, num_clients: int):
    """
    Load ALL client data vào RAM (CPU tensors).
    KHÔNG load lên GPU - sẽ load lên GPU khi train.
    
    Returns:
        client_data: List[Dict] với 'X_train', 'y_train' là CPU tensors
        test_data: Dict với 'X_test', 'y_test' là CPU tensors  
        input_shape, num_classes
    """
    print("\n" + "="*80)
    print("📥 LOADING DATA INTO RAM (CPU)")
    print("="*80)
    
    client_data = []
    all_labels = []
    
    for cid in range(num_clients):
        path = os.path.join(data_dir, f"client_{cid}_train.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        
        data = np.load(path)
        X_train = data['X_train'].astype(np.float32)
        y_train = data['y_train'].astype(np.int64)
        
        # GIỮ TRÊN CPU - không .to(device)
        X_train_t = torch.from_numpy(X_train)  # CPU tensor
        y_train_t = torch.from_numpy(y_train)  # CPU tensor
        
        client_data.append({
            'X_train': X_train_t,
            'y_train': y_train_t,
            'num_samples': len(y_train)
        })
        
        all_labels.append(y_train)
        
        if (cid + 1) % 100 == 0 or cid == num_clients - 1:
            print(f"  Loaded client {cid+1}/{num_clients}: {len(y_train):,} samples")
    
    # Load global test data - cũng giữ trên CPU
    test_path = os.path.join(data_dir, "global_test_data.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing: {test_path}")
    
    test_npz = np.load(test_path)
    X_test = test_npz['X_test'].astype(np.float32)
    y_test = test_npz['y_test'].astype(np.int64)
    
    test_data = {
        'X_test': torch.from_numpy(X_test),   # CPU
        'y_test': torch.from_numpy(y_test),   # CPU
        'num_samples': len(y_test)
    }
    
    print(f"\n  ✓ Loaded global test: {len(y_test):,} samples")
    
    # Detect shape & classes
    input_shape = (client_data[0]['X_train'].shape[1],)
    all_labels_np = np.concatenate(all_labels)
    num_classes = int(len(np.unique(all_labels_np)))
    
    total_train = sum(c['num_samples'] for c in client_data)
    print(f"\n  📊 input_shape: {input_shape}")
    print(f"  📊 num_classes: {num_classes}")
    print(f"  📊 total_train: {total_train:,}")
    print(f"  📊 total_test:  {len(y_test):,}")
    print(f"\n  ✅ All data loaded into RAM (CPU)!")
    print("="*80)
    
    return client_data, test_data, input_shape, num_classes


# =============================================================================
# FEDERATED CLIENT - Multi-GPU Compatible
# =============================================================================
class FederatedClientMultiGPU:
    """
    Client hỗ trợ Multi-GPU:
    - Data giữ trên CPU
    - Khi train, load batch lên GPU được chỉ định
    """
    
    def __init__(self, client_id: int, X_train: torch.Tensor, y_train: torch.Tensor):
        self.client_id = client_id
        self.X_train = X_train  # CPU tensor
        self.y_train = y_train  # CPU tensor
        self.num_samples = len(y_train)
        
        # Model sẽ được set sau khi assign GPU
        self.model = None
        self.device = None
    
    def setup_for_gpu(self, model: nn.Module, device: str):
        """Setup client để train trên GPU cụ thể"""
        self.model = model
        self.device = device
        self.use_amp = ("cuda" in device)
    
    def _amp_ctx(self):
        return (
            torch_autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp else contextlib.nullcontext()
        )
    
    def _create_batches(self, batch_size: int):
        """Tạo batches và move lên GPU khi cần"""
        indices = torch.randperm(self.num_samples)
        for i in range(0, self.num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            # Load batch lên GPU
            X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
            y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)
            yield X_batch, y_batch
    
    def train_fedavg(self, epochs: int, batch_size: int, lr: float):
        """Train với FedAvg"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.use_amp)
        
        total_loss = 0.0
        total_samples = 0
        
        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    loss = criterion(out, y_batch)
                
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs
        
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict((k, v.cpu().clone()) for k, v in self.model.state_dict().items())
        }
    
    def train_fedprox(self, epochs: int, batch_size: int, global_params: OrderedDict,
                      mu: float, lr: float):
        """Train với FedProx"""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.use_amp)
        
        # Pre-move global params to device
        global_params_device = {k: v.to(self.device) for k, v in global_params.items()}
        
        total_loss = 0.0
        total_samples = 0
        
        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    ce_loss = criterion(out, y_batch)
                    
                    # Proximal term
                    prox = 0.0
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in global_params_device:
                            prox += torch.sum((param - global_params_device[name])**2)
                    
                    loss = ce_loss + (mu / 2.0) * prox
                
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs
        
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict((k, v.cpu().clone()) for k, v in self.model.state_dict().items())
        }
    
    def train_fedplus(self, epochs: int, batch_size: int, global_params: OrderedDict,
                      mu: float, lr: float):
        """Train với Fed+"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.use_amp)
        
        # Pre-move global params
        global_params_device = {k: v.to(self.device) for k, v in global_params.items()}
        
        # Theta for Fed+
        theta = 1.0 / (1.0 + lr * mu)
        
        total_loss = 0.0
        total_samples = 0
        
        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    loss = criterion(out, y_batch)
                
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Fed+ correction
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in global_params_device:
                            param.data = theta * param.data + (1.0 - theta) * global_params_device[name]
                
                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs
        
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict((k, v.cpu().clone()) for k, v in self.model.state_dict().items())
        }


# =============================================================================
# MULTI-GPU TRAINING LOGIC
# =============================================================================
def train_clients_on_gpu(gpu_id: int, clients: List[FederatedClientMultiGPU],
                          global_params: OrderedDict, config: Dict,
                          results_dict: Dict, algorithm: str,
                          use_cpu: bool = False):
    """
    Train một nhóm clients trên 1 GPU cụ thể (hoặc CPU).
    Hàm này chạy trong 1 thread riêng.
    
    Args:
        gpu_id: GPU index (0, 1, 2, ...)
        use_cpu: Nếu True, dùng CPU thay vì GPU
    """
    if use_cpu:
        device = "cpu"
        device_name = "CPU"
    else:
        device = f"cuda:{gpu_id}"
        device_name = f"GPU {gpu_id}"
    
    gpu_start = time.time()
    
    # Tạo model cho device này
    model = CNN_GRU_Model(config["input_shape"], config["num_classes"]).to(device)
    
    print(f"      [{device_name}] Starting {len(clients)} clients...")
    
    for idx, client in enumerate(clients):
        client_start = time.time()
        
        # Load global params vào model
        model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        
        # Setup client
        client.setup_for_gpu(model, device)
        
        # Train theo algorithm
        if algorithm == "fedavg":
            result = client.train_fedavg(
                config["local_epochs"], config["batch_size"], config["learning_rate"]
            )
        elif algorithm == "fedprox":
            result = client.train_fedprox(
                config["local_epochs"], config["batch_size"], 
                global_params, config["mu"], config["learning_rate"]
            )
        elif algorithm == "fedavgm":
            # FedAvgM uses same client training as FedAvg
            result = client.train_fedavg(
                config["local_epochs"], config["batch_size"], config["learning_rate"]
            )
        elif algorithm == "fedplus":
            result = client.train_fedplus(
                config["local_epochs"], config["batch_size"],
                global_params, config["mu"], config["learning_rate"]
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        client_time = time.time() - client_start
        print(f"      [{device_name}] Client {client.client_id} done in {client_time:.2f}s (loss: {result['loss']:.4f})")
        
        results_dict[client.client_id] = result
    
    gpu_time = time.time() - gpu_start
    print(f"      [{device_name}] ✓ All {len(clients)} clients done in {gpu_time:.2f}s")
    
    # Clear GPU memory (only if using GPU)
    del model
    if not use_cpu:
        torch.cuda.empty_cache()


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


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_federated_multigpu(server: FederatedServerMultiGPU, config: Dict):
    """Main training loop"""
    R = config["num_rounds"]
    eval_every = config["eval_every"]
    history = server.history
    
    for ridx in tqdm(range(R), desc="Global Rounds"):
        print(f"\n{'='*60}")
        print(f"ROUND {ridx+1}/{R}")
        print(f"{'='*60}")
        
        # Train
        r_res = server.train_round(verbose=True)
        
        # Always record train loss
        history["train_loss"].append(r_res["train_loss"])
        
        # Evaluate
        if (ridx + 1) % eval_every == 0:
            print("\n  📊 Evaluating global model...")
            metrics = server.evaluate_global()
            
            history["test_loss"].append(metrics["loss"])
            history["test_accuracy"].append(metrics["accuracy"])
            history["test_f1_macro"].append(metrics["f1_macro"])
            history["test_f1_weighted"].append(metrics["f1_weighted"])
            history["test_precision_macro"].append(metrics["precision_macro"])
            history["test_recall_macro"].append(metrics["recall_macro"])
            history["test_auc_macro"].append(metrics.get("auc_macro_ovr"))
            
            print(f"\n{'='*60}")
            print(f"📊 METRICS SUMMARY - Round {ridx+1}")
            print(f"{'='*60}")
            print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
            print(f"  F1 (macro):         {metrics['f1_macro']*100:.2f}%")
            print(f"  F1 (weighted):      {metrics['f1_weighted']*100:.2f}%")
            print(f"  Precision (macro):  {metrics['precision_macro']*100:.2f}%")
            print(f"  Recall (macro):     {metrics['recall_macro']*100:.2f}%")
            if metrics.get("auc_macro_ovr"):
                print(f"  AUC (macro OvR):    {metrics['auc_macro_ovr']*100:.2f}%")
            print(f"{'='*60}")
    
    return history


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*80)
    print("🚀 FEDERATED LEARNING WITH MULTI-GPU SUPPORT")
    print("="*80)
    
    # Load data vào RAM
    client_data, test_data, input_shape, num_classes = load_all_client_data_to_ram(
        CONFIG["data_dir"], CONFIG["num_clients"]
    )
    
    CONFIG["input_shape"] = input_shape
    CONFIG["num_classes"] = num_classes
    
    print(f"\nConfig: {json.dumps(CONFIG, indent=2, default=str)}")
    
    # Tạo clients
    clients = []
    for cid in range(CONFIG["num_clients"]):
        c = FederatedClientMultiGPU(
            client_id=cid,
            X_train=client_data[cid]['X_train'],
            y_train=client_data[cid]['y_train']
        )
        clients.append(c)
    
    print(f"\n✓ Created {len(clients)} clients")
    
    # Tạo server
    server = FederatedServerMultiGPU(clients, test_data, CONFIG)
    
    # Train
    start_time = datetime.now()
    history = train_federated_multigpu(server, CONFIG)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Training complete! Duration: {duration:.2f}s ({duration/60:.2f} min)")
    
    # Save
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    algo = CONFIG["algorithm"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = os.path.join(CONFIG["output_dir"], f"{algo}_model_{ts}.pth")
    torch.save(server.global_model.state_dict(), model_path)
    print(f"💾 Saved model: {model_path}")
    
    hist_path = os.path.join(CONFIG["output_dir"], f"{algo}_history_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Saved history: {hist_path}")


if __name__ == "__main__":
    main()
