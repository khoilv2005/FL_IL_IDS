"""
Data Loading Functions for Federated Learning
"""

import os
import numpy as np
import torch


def load_all_client_data_to_ram(data_dir: str, num_clients: int):
    """
    Load ALL client data vào RAM (CPU tensors).
    KHÔNG load lên GPU - sẽ load lên GPU khi train.
    
    Args:
        data_dir: Directory containing client data files
        num_clients: Number of clients to load
    
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
