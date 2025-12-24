"""
STEP 2 VERSION 3: Create Non-IID Federated Splits (Test Split First)
====================================================================
Logic MỚI: Tách test DATA TRƯỚC, rồi mới chia train cho clients

Workflow:
1. Load ALL data từ chunks
2. Train/Test split TRƯỚC (70/30) → Test data độc lập
3. Chia CHỈ phần TRAIN cho clients theo Non-IID
4. Save client train data
5. Save global test data

So với Version 2:
- Version 2: Chia clients trước → mỗi client chia 70/30
- Version 3: Chia 70/30 trước → chỉ chia 70% cho clients
"""

import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'chunks_dir': './data/preprocessed_chunks',
    'output_dir': './data/federated_splits/1000-clients',
    'num_clients': 1000,
    'non_iid_alpha': 0.5,
    'test_size': 0.3,
}

# ============================================================================
# STEP 1: LOAD ALL DATA
# ============================================================================

def load_all_data(chunks_dir):
    """
    Load toàn bộ data từ chunks
    """
    logger.info("="*80)
    logger.info("STEP 1: LOADING ALL DATA FROM CHUNKS")
    logger.info("="*80)

    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.npz")))

    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No chunks found in {chunks_dir}")

    logger.info(f"Found {len(chunk_files)} chunks")

    all_X = []
    all_y = []

    for i, chunk_file in enumerate(chunk_files):
        logger.info(f"  Loading chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
        data = np.load(chunk_file)
        all_X.append(data['X'])
        all_y.append(data['y'])

        # Clear memory
        del data
        gc.collect()

    # Concatenate all chunks
    logger.info("\nConcatenating all chunks...")
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    # Clear memory
    del all_X, all_y
    gc.collect()

    logger.info(f"\n✓ Loaded all data:")
    logger.info(f"  Shape: {X_all.shape}")
    logger.info(f"  Total samples: {len(y_all):,}")
    logger.info(f"  Number of classes: {len(np.unique(y_all))}")
    logger.info(f"  Label distribution: {np.bincount(y_all)}")

    return X_all, y_all

# ============================================================================
# STEP 2: TRAIN/TEST SPLIT FIRST
# ============================================================================

def split_train_test_first(X_all, y_all, test_size):
    """
    Chia train/test TRƯỚC KHI chia clients
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAIN/TEST SPLIT (BEFORE CLIENT SPLIT)")
    logger.info("="*80)
    logger.info(f"Test size: {test_size} ({test_size*100:.0f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=42,
        stratify=y_all
    )

    logger.info(f"\n✓ Split complete:")
    logger.info(f"  Train: {len(y_train):,} samples ({(1-test_size)*100:.0f}%)")
    logger.info(f"  Test:  {len(y_test):,} samples ({test_size*100:.0f}%)")
    logger.info(f"\n  Train distribution: {np.bincount(y_train)}")
    logger.info(f"  Test distribution:  {np.bincount(y_test)}")

    # Clear memory
    del X_all, y_all
    gc.collect()

    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 3: CREATE NON-IID CLIENT SPLIT (TRAIN ONLY)
# ============================================================================

def create_non_iid_client_indices(y_train, num_clients, alpha):
    """
    Tạo Non-IID indices cho clients (CHỈ TRAIN DATA)
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CREATE NON-IID CLIENT SPLIT (TRAIN DATA ONLY)")
    logger.info("="*80)
    logger.info(f"Number of clients: {num_clients}")
    logger.info(f"Dirichlet alpha: {alpha}")
    logger.info(f"Train samples to split: {len(y_train):,}")

    num_classes = len(np.unique(y_train))
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        # Get indices for this class
        class_indices = np.where(y_train == class_id)[0]
        np.random.shuffle(class_indices)

        # Dirichlet distribution
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
        proportions = np.concatenate([[0], proportions])

        # Split according to proportions
        for client_id in range(num_clients):
            start = proportions[client_id]
            end = proportions[client_id + 1]
            client_indices[client_id].extend(class_indices[start:end])

    # Shuffle each client's indices
    for client_id in range(num_clients):
        client_indices[client_id] = np.array(client_indices[client_id])
        np.random.shuffle(client_indices[client_id])
        logger.info(f"  Client {client_id}: {len(client_indices[client_id]):,} samples")

    logger.info("\n✓ Non-IID client split created")

    return client_indices

# ============================================================================
# STEP 4: SAVE CLIENT DATA & GLOBAL TEST
# ============================================================================

def save_client_and_test_data(X_train, y_train, X_test, y_test, client_indices, output_dir):
    """
    Save client train data và global test data
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 4: SAVING CLIENT DATA & GLOBAL TEST")
    logger.info("="*80)

    os.makedirs(output_dir, exist_ok=True)

    num_clients = len(client_indices)
    num_classes = len(np.unique(y_train))
    client_stats = []

    # Save each client's train data
    for client_id in range(num_clients):
        indices = client_indices[client_id]

        X_client = X_train[indices]
        y_client = y_train[indices]

        # Save client train data
        save_path = os.path.join(output_dir, f"client_{client_id}_train.npz")
        np.savez_compressed(
            save_path,
            X_train=X_client,
            y_train=y_client
        )

        file_size_mb = os.path.getsize(save_path) / (1024**2)

        logger.info(f"\nClient {client_id}:")
        logger.info(f"  Samples: {len(y_client):,}")
        logger.info(f"  File: {save_path}")
        logger.info(f"  Size: {file_size_mb:.2f} MB")

        # Collect stats
        train_dist = np.bincount(y_client, minlength=num_classes)

        client_stats.append({
            'client_id': client_id,
            'train_samples': len(y_client),
            'test_samples': 0,  # No test per client
            'total_samples': len(y_client),
            'train_dist': train_dist,
            'test_dist': np.zeros(num_classes, dtype=int),  # Empty
            'train_file_size_mb': file_size_mb,
            'train_memory_mb': X_client.nbytes / (1024**2),
            'test_memory_mb': 0
        })

        # Clear memory
        del X_client, y_client
        gc.collect()

    # Save global test data
    logger.info(f"\n{'='*80}")
    logger.info("Saving global test data...")
    test_save_path = os.path.join(output_dir, "global_test_data.npz")
    np.savez_compressed(
        test_save_path,
        X_test=X_test,
        y_test=y_test
    )

    test_file_size_mb = os.path.getsize(test_save_path) / (1024**2)
    logger.info(f"✓ Saved global test data:")
    logger.info(f"  File: {test_save_path}")
    logger.info(f"  Samples: {len(y_test):,}")
    logger.info(f"  Size: {test_file_size_mb:.2f} MB")

    return client_stats, test_file_size_mb

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("STEP 2 VERSION 3: NON-IID SPLIT (TEST SPLIT FIRST)")
    logger.info("="*80)
    logger.info("\nWorkflow:")
    logger.info("  1. Load all data from chunks")
    logger.info("  2. Train/Test split FIRST (70/30)")
    logger.info("  3. Create Non-IID split for TRAIN data only")
    logger.info("  4. Save client train data")
    logger.info("  5. Save global test data")
    logger.info("\nMemory requirement: ~4-8 GB RAM")
    logger.info("="*80)

    config = CONFIG

    logger.info(f"\nConfiguration:")
    logger.info(f"  Chunks directory: {config['chunks_dir']}")
    logger.info(f"  Output directory: {config['output_dir']}")
    logger.info(f"  Number of clients: {config['num_clients']}")
    logger.info(f"  Non-IID alpha: {config['non_iid_alpha']}")
    logger.info(f"  Test size: {config['test_size']}")

    # Step 1: Load all data
    X_all, y_all = load_all_data(config['chunks_dir'])

    # Step 2: Train/Test split FIRST
    X_train, X_test, y_train, y_test = split_train_test_first(
        X_all, y_all,
        test_size=config['test_size']
    )

    # Step 3: Create Non-IID client split (train only)
    client_indices = create_non_iid_client_indices(
        y_train=y_train,
        num_clients=config['num_clients'],
        alpha=config['non_iid_alpha']
    )

    # Step 4: Save client data & global test
    client_stats, test_file_size_mb = save_client_and_test_data(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        client_indices=client_indices,
        output_dir=config['output_dir']
    )

    logger.info("\n" + "="*80)
    logger.info("✓ STEP 2 VERSION 3 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutput:")
    logger.info(f"  Directory: {config['output_dir']}")
    logger.info(f"  Client files: client_0_train.npz to client_{config['num_clients']-1}_train.npz")
    logger.info(f"  Test file: global_test_data.npz")
    logger.info("\nNext step:")
    logger.info("  Use client_*_train.npz for federated training")
    logger.info("  Use global_test_data.npz for global evaluation")
    logger.info("\nNote: Test data is INDEPENDENT from client splits!")
    logger.info("\nNote: Use non_iid_visualize.py for visualization")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
