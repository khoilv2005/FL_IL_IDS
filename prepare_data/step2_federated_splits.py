"""
STEP 2: Federated Data Splitting (Refactored)
==============================================
Chia dữ liệu cho federated learning với incremental learning support.

Workflow:
1. Load raw data từ chunks
2. Train/Test split
3. Fit scaler on TRAIN only (no data leak)
4. Define task structure (incremental learning)
5. Allocate clients to tasks
6. Distribute data to clients
7. Save data + metadata

NO DATA LEAK: Test data không ảnh hưởng đến scaler
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import glob
import logging
import pickle
import json
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for federated splitting."""
    # Paths
    chunks_dir: str = './data/raw_chunks'
    output_dir: str = './data/federated_splits/100-clients'
    
    # Client settings
    num_clients: int = 100
    
    # Incremental Learning
    base_classes: int = 10
    classes_per_task: int = 6
    
    # Client participation
    old_client_data_prob: float = 1.0  # Prob that old clients receive new task data
    
    # Data distribution - Non-IID settings
    distribution_strategy: str = "dirichlet"  # "iid", "dirichlet"
    dirichlet_alpha: float = 5.0  # Moderate alpha
    class_sparsity: float = 0.7  # 0.7 density => ~30-40% empty spots + client activation logic ~ 50%
    min_clients_per_class: int = 2
    
    # Balancing
    max_samples_per_class: int = 0  # 0 = no limit
    min_client_ratio: float = 0.0  # 0.0 = No minimum constraint (natural distribution)
    
    # Split
    test_size: float = 0.3
    random_seed: Optional[int] = None





# ============================================================================
# TASK STRUCTURE - Định nghĩa incremental tasks
# ============================================================================

class TaskStructure:
    """
    Định nghĩa cấu trúc task cho Class Incremental Learning.
    
    Task 0: base_classes (ví dụ: 10 classes đầu)
    Task 1+: classes_per_task mỗi task
    """
    
    def __init__(self, total_classes: int, base_classes: int, classes_per_task: int):
        self.total_classes = total_classes
        self.base_classes = base_classes
        self.classes_per_task = classes_per_task
        
        # Calculate number of tasks
        remaining = total_classes - base_classes
        self.num_tasks = 1 + max(0, (remaining + classes_per_task - 1) // classes_per_task)
        
        # Generate task -> classes mapping
        self._task_classes = self._generate_task_classes()
        
        logger.info(f"TaskStructure: {self.num_tasks} tasks for {total_classes} classes")
    
    def _generate_task_classes(self) -> Dict[int, List[int]]:
        """Generate mapping: task_id -> list of class indices."""
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
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get classes introduced in a specific task."""
        return self._task_classes.get(task_id, [])
    
    def get_seen_classes(self, task_id: int) -> List[int]:
        """Get all classes seen up to and including task_id."""
        seen = []
        for t in range(task_id + 1):
            seen.extend(self._task_classes.get(t, []))
        return seen
    
    def to_dict(self) -> dict:
        """Export as dictionary for saving."""
        return {
            "total_classes": self.total_classes,
            "base_classes": self.base_classes,
            "classes_per_task": self.classes_per_task,
            "num_tasks": self.num_tasks,
            "task_classes": {str(k): v for k, v in self._task_classes.items()}
        }
    
    def __repr__(self):
        return f"TaskStructure(tasks={self.num_tasks}, classes={self.total_classes})"


# ============================================================================
# CLIENT ALLOCATOR - Xác định client tham gia task nào
# ============================================================================

class ClientAllocator:
    """
    Xác định client nào tham gia task nào.
    
    Logic:
    - Chia đều clients cho các tasks
    - Clients mới join ở mỗi task
    - Clients cũ có thể nhận data mới với xác suất old_client_data_prob
    """
    
    def __init__(
        self, 
        num_clients: int, 
        num_tasks: int,
        old_client_data_prob: float = 1.0,
        random_seed: int = 42
    ):
        self.num_clients = num_clients
        self.num_tasks = num_tasks
        self.old_client_data_prob = old_client_data_prob
        self.rng = np.random.default_rng(random_seed)
        
        # Allocate
        self._client_join_task: Dict[int, int] = {}  # client_id -> task joined
        self._task_active_clients: Dict[int, List[int]] = {}  # task_id -> active clients
        self._task_new_clients: Dict[int, List[int]] = {}  # task_id -> new clients
        
        self._allocate()
    
    def _allocate(self):
        """Perform allocation."""
        clients_per_task = self.num_clients // self.num_tasks
        available = list(range(self.num_clients))
        self.rng.shuffle(available)
        
        for task_id in range(self.num_tasks):
            # Get new clients for this task
            num_new = min(clients_per_task, len(available))
            if task_id == self.num_tasks - 1:
                # Last task gets all remaining
                num_new = len(available)
            
            new_clients = available[:num_new]
            available = available[num_new:]
            
            # Record join time
            for cid in new_clients:
                self._client_join_task[cid] = task_id
            
            self._task_new_clients[task_id] = list(new_clients)
            
            # Determine active clients (new + old with probability)
            active_clients = list(new_clients)
            if task_id > 0:
                for cid in range(self.num_clients):
                    if cid in self._client_join_task and self._client_join_task[cid] < task_id:
                        if self.rng.random() < self.old_client_data_prob:
                            active_clients.append(cid)
            
            self._task_active_clients[task_id] = active_clients
            
            logger.info(f"  Task {task_id}: {len(new_clients)} new, {len(active_clients)} active")
    
    def get_new_clients(self, task_id: int) -> List[int]:
        """Get clients that joined at this task."""
        return self._task_new_clients.get(task_id, [])
    
    def get_active_clients(self, task_id: int) -> List[int]:
        """Get all active clients for this task."""
        return self._task_active_clients.get(task_id, [])
    
    def get_client_join_task(self, client_id: int) -> int:
        """Get the task when client joined."""
        return self._client_join_task.get(client_id, -1)
    
    def to_dict(self) -> dict:
        """Export as dictionary for saving."""
        return {
            "num_clients": self.num_clients,
            "num_tasks": self.num_tasks,
            "old_client_data_prob": self.old_client_data_prob,
            "client_join_task": self._client_join_task,
            "task_new_clients": {str(k): v for k, v in self._task_new_clients.items()},
            "task_active_clients": {str(k): v for k, v in self._task_active_clients.items()}
        }


# ============================================================================
# DATA DISTRIBUTOR - Chia data cho clients
# ============================================================================

class DataDistributor:
    """
    Chia data theo các chiến lược khác nhau.
    
    Strategies:
    - "iid": Chia đều ngẫu nhiên
    - "dirichlet": Non-IID theo Dirichlet distribution
    """
    
    def __init__(
        self,
        strategy: str = "dirichlet",
        alpha: float = 0.5,
        class_sparsity: float = 0.6,
        min_clients_per_class: int = 2,
        random_seed: int = 42
    ):
        self.strategy = strategy
        self.alpha = alpha
        self.class_sparsity = class_sparsity
        self.min_clients_per_class = min_clients_per_class
        self.rng = np.random.default_rng(random_seed)
    
    def distribute(
        self,
        y_train: np.ndarray,
        task_classes: List[int],
        active_clients: List[int]
    ) -> Dict[int, List[int]]:
        """
        Distribute data indices to clients.
        
        Returns:
            Dict mapping client_id -> list of indices
        """
        if self.strategy == "iid":
            return self._distribute_iid(y_train, task_classes, active_clients)
        elif self.strategy == "dirichlet":
            return self._distribute_dirichlet(y_train, task_classes, active_clients)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _distribute_iid(
        self, 
        y_train: np.ndarray, 
        task_classes: List[int],
        active_clients: List[int]
    ) -> Dict[int, List[int]]:
        """IID distribution - chia đều ngẫu nhiên."""
        # Get all indices for task classes
        all_indices = []
        for c in task_classes:
            idx = np.where(y_train == c)[0]
            all_indices.extend(idx)
        
        all_indices = np.array(all_indices)
        self.rng.shuffle(all_indices)
        
        # Split equally
        splits = np.array_split(all_indices, len(active_clients))
        
        result = {}
        for i, cid in enumerate(active_clients):
            result[cid] = list(splits[i])
        
        return result
    
    def _distribute_dirichlet(
        self,
        y_train: np.ndarray,
        task_classes: List[int],
        active_clients: List[int]
    ) -> Dict[int, List[int]]:
        """
        Non-IID Dirichlet distribution.
        
        Cho mỗi class:
        1. Chọn subset clients (class_sparsity)
        2. Chia data theo Dirichlet(alpha)
        """
        result = {cid: [] for cid in active_clients}
        
        for c in task_classes:
            idx = np.where(y_train == c)[0]
            if len(idx) == 0:
                continue
            
            # Shuffle indices
            self.rng.shuffle(idx)
            
            # Select subset of clients for this class
            n_active = len(active_clients)
            n_recv = max(
                self.min_clients_per_class,
                int(np.ceil(self.class_sparsity * n_active))
            )
            n_recv = min(n_recv, n_active)
            
            recv_clients = self.rng.choice(active_clients, n_recv, replace=False)
            
            # Dirichlet distribution
            alpha_eff = max(0.05, self.alpha / np.sqrt(n_recv))
            proportions = self.rng.dirichlet([alpha_eff] * n_recv)
            counts = (proportions * len(idx)).astype(int)
            
            # Fix remainder - GUARANTEE all data is distributed
            remainder = len(idx) - counts.sum()
            if remainder > 0:
                # Add remaining samples to random clients
                add_to = self.rng.choice(n_recv, remainder, replace=True)
                for j in add_to:
                    counts[j] += 1
            elif remainder < 0:
                # Remove excess (shouldn't happen normally, but safety check)
                while counts.sum() > len(idx):
                    nonzero = np.where(counts > 0)[0]
                    if len(nonzero) == 0:
                        break
                    remove_from = self.rng.choice(nonzero)
                    counts[remove_from] -= 1
            
            # Verify: assert counts.sum() == len(idx)
            assert counts.sum() == len(idx), f"Data loss! {counts.sum()} != {len(idx)}"
            
            # Split and assign
            split_points = np.cumsum(counts)[:-1]
            chunks = np.split(idx, split_points)
            
            for j, cid in enumerate(recv_clients):
                result[cid].extend(chunks[j].tolist())
        
        return result


# ============================================================================
# DATA LOADER & SCALER
# ============================================================================

def load_raw_data(chunks_dir: str, max_samples_per_class: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw data from chunks."""
    logger.info("=" * 60)
    logger.info("Loading raw data from chunks")
    logger.info("=" * 60)
    
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.npz")))
    
    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")
    
    logger.info(f"Found {len(chunk_files)} chunks")
    
    all_X, all_y = [], []
    for i, chunk_file in enumerate(chunk_files):
        logger.info(f"  Loading chunk {i+1}/{len(chunk_files)}")
        data = np.load(chunk_file)
        all_X.append(data['X'])
        all_y.append(data['y'])
        del data
        gc.collect()
    
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    del all_X, all_y
    gc.collect()
    
    # Balance if configured
    if max_samples_per_class > 0:
        logger.info(f"Balancing: max {max_samples_per_class} per class")
        classes, counts = np.unique(y_all, return_counts=True)
        keep_indices = []
        
        for c in classes:
            idx = np.where(y_all == c)[0]
            if len(idx) > max_samples_per_class:
                selected = np.random.choice(idx, max_samples_per_class, replace=False)
                keep_indices.extend(selected)
            else:
                keep_indices.extend(idx)
        
        keep_indices = np.array(keep_indices)
        np.random.shuffle(keep_indices)
        X_all = X_all[keep_indices]
        y_all = y_all[keep_indices]
    
    logger.info(f"Loaded: {X_all.shape}, {len(np.unique(y_all))} classes")
    return X_all, y_all


def fit_and_scale(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    output_dir: str
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on train only, transform both."""
    logger.info("=" * 60)
    logger.info("Fitting scaler on TRAIN only (no data leak)")
    logger.info("=" * 60)
    
    # Reshape if needed
    if X_train.ndim == 3 and X_train.shape[2] == 1:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
    
    # Compute bounds from train
    lower = np.percentile(X_train, 0.1)
    upper = np.percentile(X_train, 99.9)
    X_train = np.clip(X_train, lower, upper)
    X_test = np.clip(X_test, lower, upper)
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train).astype(np.float16)
    X_test_scaled = scaler.transform(X_test).astype(np.float16)
    
    # Add channel dimension
    X_train_scaled = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)
    
    # Save scaler
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# FEDERATED SPLITTER - Main Orchestrator
# ============================================================================

class FederatedSplitter:
    """
    Main class điều phối toàn bộ quá trình chia data.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.task_structure: Optional[TaskStructure] = None
        self.client_allocator: Optional[ClientAllocator] = None
        self.data_distributor: Optional[DataDistributor] = None
    
    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 80)
        logger.info("FEDERATED DATA SPLITTING")
        logger.info("=" * 80)
        
        # Step 1: Load raw data
        X_all, y_all = load_raw_data(
            self.config.chunks_dir,
            self.config.max_samples_per_class
        )
        
        total_classes = len(np.unique(y_all))
        logger.info(f"Total classes: {total_classes}")
        
        # Step 2: Train/Test split
        logger.info("\nSplitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y_all
        )
        del X_all, y_all
        gc.collect()
        
        logger.info(f"Train: {len(y_train):,}, Test: {len(y_test):,}")
        
        # Step 3: Scale
        X_train_scaled, X_test_scaled, _ = fit_and_scale(
            X_train, X_test, self.config.output_dir
        )
        del X_train, X_test
        gc.collect()
        
        # Step 4: Create task structure
        logger.info("\nCreating task structure...")
        self.task_structure = TaskStructure(
            total_classes=total_classes,
            base_classes=self.config.base_classes,
            classes_per_task=self.config.classes_per_task
        )
        
        for t in range(self.task_structure.num_tasks):
            classes = self.task_structure.get_task_classes(t)
            logger.info(f"  Task {t}: classes {classes}")
        
        # Step 5: Allocate clients
        logger.info("\nAllocating clients to tasks...")
        self.client_allocator = ClientAllocator(
            num_clients=self.config.num_clients,
            num_tasks=self.task_structure.num_tasks,
            old_client_data_prob=self.config.old_client_data_prob,
            random_seed=self.config.random_seed
        )
        
        # Step 6: Create data distributor
        self.data_distributor = DataDistributor(
            strategy=self.config.distribution_strategy,
            alpha=self.config.dirichlet_alpha,
            class_sparsity=self.config.class_sparsity,
            min_clients_per_class=self.config.min_clients_per_class,
            random_seed=self.config.random_seed
        )
        
        # Step 7: Distribute data for each task
        logger.info("\nDistributing data to clients...")
        client_indices = {cid: [] for cid in range(self.config.num_clients)}
        
        for task_id in range(self.task_structure.num_tasks):
            task_classes = self.task_structure.get_task_classes(task_id)
            active_clients = self.client_allocator.get_active_clients(task_id)
            
            task_distribution = self.data_distributor.distribute(
                y_train, task_classes, active_clients
            )
            
            for cid, indices in task_distribution.items():
                client_indices[cid].extend(indices)
            
            logger.info(f"  Task {task_id}: distributed to {len(active_clients)} clients")
        
        # Step 8: Enforce minimum samples
        client_indices = self._enforce_min_samples(
            client_indices, len(y_train)
        )
        
        # Step 9: Save data
        self._save_data(X_train_scaled, y_train, X_test_scaled, y_test, client_indices)
        
        # Step 10: Save metadata
        self._save_metadata(y_train, client_indices)
        
        logger.info("\n" + "=" * 80)
        logger.info("COMPLETED!")
        logger.info("=" * 80)
    
    def _enforce_min_samples(
        self, 
        client_indices: Dict[int, List[int]], 
        total_train: int
    ) -> Dict[int, List[int]]:
        """Ensure all clients have minimum samples."""
        avg = total_train / self.config.num_clients
        min_samples = max(1, int(np.ceil(avg * self.config.min_client_ratio)))
        
        counts = {cid: len(idx) for cid, idx in client_indices.items()}
        min_count = min(counts.values())
        
        if min_count >= min_samples:
            logger.info(f"All clients >= {min_samples} samples")
            return client_indices
        
        logger.info(f"Balancing: min_samples={min_samples}")
        
        # Move samples from donors to receivers
        donors = [cid for cid, c in counts.items() if c > min_samples]
        receivers = [cid for cid, c in counts.items() if c < min_samples]
        
        rng = np.random.default_rng(self.config.random_seed)
        
        for rid in receivers:
            need = min_samples - len(client_indices[rid])
            for did in donors:
                available = len(client_indices[did]) - min_samples
                if available <= 0:
                    continue
                take = min(need, available)
                moved = client_indices[did][-take:]
                client_indices[did] = client_indices[did][:-take]
                client_indices[rid].extend(moved)
                need -= take
                if need <= 0:
                    break
        
        return client_indices
    
    def _save_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_indices: Dict[int, List[int]]
    ):
        """Save client data and test data."""
        logger.info("\nSaving data...")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        for cid, indices in client_indices.items():
            if len(indices) == 0:
                continue
            
            indices = np.array(indices, dtype=np.int64)
            X_client = X_train[indices]
            y_client = y_train[indices]
            
            path = os.path.join(self.config.output_dir, f"client_{cid}_train.npz")
            np.savez_compressed(path, X_train=X_client, y_train=y_client)
            
            size_mb = os.path.getsize(path) / (1024 ** 2)
            logger.info(f"  Client {cid}: {len(y_client):,} samples, {size_mb:.1f}MB")
        
        # Save test data
        test_path = os.path.join(self.config.output_dir, "global_test_data.npz")
        np.savez_compressed(test_path, X_test=X_test, y_test=y_test)
        logger.info(f"  Test: {len(y_test):,} samples")
    
    def _save_metadata(
        self,
        y_train: np.ndarray, 
        client_indices: Dict[int, List[int]]
    ):
        """Save metadata about the split."""
        metadata = {
            "config": asdict(self.config),
            "task_structure": self.task_structure.to_dict(),
            "client_allocation": self.client_allocator.to_dict(),
            "statistics": {
                "total_train_samples": len(y_train),
                "samples_per_client": {
                    str(cid): len(idx) for cid, idx in client_indices.items()
                }
            }
        }
        
        path = os.path.join(self.config.output_dir, "metadata.json")
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"\nSaved metadata to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config(
        chunks_dir='./data/raw_chunks',
        output_dir='./data/federated_splits/10-clients',
        num_clients=10,
        base_classes=10,
        classes_per_task=6,
        distribution_strategy="dirichlet",
        dirichlet_alpha=5.0,
        class_sparsity=0.7,
        min_clients_per_class=2,
        old_client_data_prob=1.0,
        test_size=0.3,
        random_seed=None  # Random every time
    )
    
    splitter = FederatedSplitter(config)
    splitter.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
