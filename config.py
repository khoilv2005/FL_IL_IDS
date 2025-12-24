"""
Configuration for Federated Learning
=====================================
Chỉnh sửa các tham số ở đây trước khi train.
"""

CONFIG = {
    # ===================
    # DATA PATHS
    # ===================
    "data_dir": "/mnt/d/Project/ai4fids_project/prepare_data/data/federated_splits/500-clients",
    "output_dir": "./results",
    "checkpoint_dir": "./checkpoints",
    
    # ===================
    # FEDERATED LEARNING
    # ===================
    "num_clients": 500,
    "algorithm": "fedplus",  # fedavg, fedprox, fedavgm, fedplus
    
    # FedAvgM
    "server_momentum": 0.9,
    "server_lr": 1.0,
    
    # FedProx / Fed+
    "mu": 0.01,
    
    # ===================
    # MODEL (auto-detect)
    # ===================
    "input_shape": None,
    "num_classes": None,
    
    # ===================
    # TRAINING
    # ===================
    "num_rounds": 5,
    "local_epochs": 3,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    
    # ===================
    # MULTI-GPU
    # ===================
    "num_gpus": None,  # None = auto-detect
    
    # ===================
    # EVAL & SAVE
    # ===================
    "eval_every": 1,
    "save_checkpoint_every": 1,
}
