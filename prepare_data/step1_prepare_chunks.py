"""
STEP 1: Chunked Preprocessing
==============================
Script này xử lý 45M samples với RAM hạn chế bằng cách:
1. Đọc từng file CSV một (không concat)
2. Xử lý từng chunk nhỏ (100K rows)
3. Dùng StandardScaler.partial_fit() để học Mean/Std toàn cục
4. Save preprocessed chunks xuống disk

Memory-efficient: Chỉ cần ~1-2GB RAM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import glob
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': './data/IoT_Dataset_2023',
    'output_dir': './data/preprocessed_chunks',
    'chunk_size': 100000,  # Process 100K rows at a time
    'pattern': 'Merged*.csv',
}

# ============================================================================
# PASS 1: Learn Mean/Std from all data
# ============================================================================

def pass1_learn_statistics(data_dir, pattern, chunk_size):
    """
    Pass 1: Đọc tất cả data để học Mean/Std mà không load hết vào RAM
    Dùng StandardScaler.partial_fit() để học dần dần
    """
    logger.info("="*80)
    logger.info("PASS 1: LEARNING GLOBAL STATISTICS")
    logger.info("="*80)

    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    logger.info(f"Found {len(csv_files)} CSV files")

    # Initialize scaler and label encoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    all_labels = []

    total_rows = 0
    total_files = len(csv_files)

    # Outlier bounds (will be computed)
    lower_bound = None
    upper_bound = None

    logger.info("\nStep 1.1: Collecting all unique labels...")

    # First, collect all labels to fit LabelEncoder
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{total_files}] Scanning labels: {os.path.basename(csv_file)}")

        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            # Normalize column names
            chunk_df.columns = chunk_df.columns.str.lower()

            if 'label' in chunk_df.columns:
                # Filter out NaN labels
                chunk_df = chunk_df[chunk_df['label'].notna()]
                all_labels.extend(chunk_df['label'].unique())

    # Fit label encoder with all unique labels
    unique_labels = list(set(all_labels))
    label_encoder.fit(unique_labels)
    logger.info(f"✓ Found {len(unique_labels)} unique labels")
    logger.info(f"  Labels: {label_encoder.classes_}")

    logger.info("\nStep 1.2: Computing percentiles for outlier clipping...")

    # Collect samples for percentile computation (sample 0.1% of data = 45K samples)
    # This uses ~20-30 MB RAM instead of 200+ MB
    sample_data = []
    sample_target = 50000  # Fixed 50K samples (safe for low RAM)
    samples_collected = 0

    for i, csv_file in enumerate(csv_files, 1):
        if samples_collected >= sample_target:
            break

        logger.info(f"[{i}/{total_files}] Sampling: {os.path.basename(csv_file)}")

        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_df.columns = chunk_df.columns.str.lower()

            if 'label' not in chunk_df.columns:
                continue

            # Get features
            X_chunk = chunk_df.drop(['label'], axis=1).values

            # Handle NaN and inf
            X_chunk = np.nan_to_num(X_chunk, nan=0.0)
            X_chunk = np.clip(X_chunk, -1e10, 1e10)

            # Randomly sample rows (collect in batches)
            n_sample = min(len(X_chunk), sample_target - samples_collected, 1000)
            if n_sample > 0:
                indices = np.random.choice(len(X_chunk), size=n_sample, replace=False)
                sample_data.append(X_chunk[indices])
                samples_collected += n_sample

            if samples_collected >= sample_target:
                break

    # Compute percentiles
    logger.info(f"Collected {samples_collected} samples for percentile computation")
    sample_matrix = np.vstack(sample_data)
    lower_bound = np.percentile(sample_matrix, 0.1)
    upper_bound = np.percentile(sample_matrix, 99.9)

    logger.info(f"✓ Computed outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    del sample_data, sample_matrix
    import gc
    gc.collect()

    logger.info("\nStep 1.3: Learning Mean/Std with partial_fit()...")

    # Now learn scaler with partial_fit
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{total_files}] Processing: {os.path.basename(csv_file)}")

        chunk_count = 0

        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_df.columns = chunk_df.columns.str.lower()

            if 'label' not in chunk_df.columns:
                continue

            # Get features
            X_chunk = chunk_df.drop(['label'], axis=1).values

            # Handle NaN and inf
            X_chunk = np.nan_to_num(X_chunk, nan=0.0)
            X_chunk = np.clip(X_chunk, -1e10, 1e10)

            # Clip outliers
            X_chunk = np.clip(X_chunk, lower_bound, upper_bound)

            # Partial fit scaler
            scaler.partial_fit(X_chunk)

            total_rows += len(X_chunk)
            chunk_count += 1

        logger.info(f"  Processed {chunk_count} chunks, {total_rows:,} total rows so far")

    logger.info(f"\n✓ Learned statistics from {total_rows:,} samples")
    logger.info(f"  Mean: {scaler.mean_[:5]}... (first 5 features)")
    logger.info(f"  Std: {scaler.scale_[:5]}... (first 5 features)")

    return scaler, label_encoder, lower_bound, upper_bound

# ============================================================================
# PASS 2: Transform and save chunks
# ============================================================================

def pass2_transform_and_save(data_dir, pattern, chunk_size, output_dir,
                             scaler, label_encoder, lower_bound, upper_bound):
    """
    Pass 2: Transform data với scaler đã học và save chunks
    """
    logger.info("\n" + "="*80)
    logger.info("PASS 2: TRANSFORMING AND SAVING CHUNKS")
    logger.info("="*80)

    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    total_files = len(csv_files)

    chunk_id = 0
    total_saved_size = 0

    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"\n[{i}/{total_files}] Processing: {os.path.basename(csv_file)}")

        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_df.columns = chunk_df.columns.str.lower()

            if 'label' not in chunk_df.columns:
                continue

            # Filter out NaN labels (same as Pass 1)
            chunk_df = chunk_df[chunk_df['label'].notna()]

            if len(chunk_df) == 0:
                continue

            # Separate features and labels
            X_chunk = chunk_df.drop(['label'], axis=1).values
            y_chunk = chunk_df['label'].values

            # Preprocess features
            X_chunk = np.nan_to_num(X_chunk, nan=0.0)
            X_chunk = np.clip(X_chunk, -1e10, 1e10)
            X_chunk = np.clip(X_chunk, lower_bound, upper_bound)

            # Transform with learned scaler
            X_chunk = scaler.transform(X_chunk)

            # Encode labels
            y_chunk = label_encoder.transform(y_chunk)

            # Reshape for CNN-GRU: (samples, features, 1)
            X_chunk = X_chunk.reshape(X_chunk.shape[0], X_chunk.shape[1], 1)

            # Convert to float16 to save memory
            X_chunk = X_chunk.astype(np.float16)

            # Save chunk
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}.npz")
            np.savez_compressed(chunk_path, X=X_chunk, y=y_chunk)

            file_size = os.path.getsize(chunk_path) / (1024**2)
            total_saved_size += file_size

            logger.info(f"  Chunk {chunk_id:04d}: {len(X_chunk):,} samples, {file_size:.2f} MB")

            chunk_id += 1

    logger.info(f"\n✓ Saved {chunk_id} chunks")
    logger.info(f"  Total size: {total_saved_size:.2f} MB ({total_saved_size/1024:.2f} GB)")

    return chunk_id

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("STEP 1: CHUNKED PREPROCESSING")
    logger.info("="*80)
    logger.info("\nThis script will:")
    logger.info("  1. Learn Mean/Std from all 45M samples (no full load)")
    logger.info("  2. Transform data in chunks")
    logger.info("  3. Save preprocessed chunks to disk")
    logger.info("\nMemory requirement: ~1-2 GB RAM")
    logger.info("="*80)

    config = CONFIG

    logger.info(f"\nConfiguration:")
    logger.info(f"  Data directory: {config['data_dir']}")
    logger.info(f"  Output directory: {config['output_dir']}")
    logger.info(f"  Chunk size: {config['chunk_size']:,} rows")
    logger.info(f"  Pattern: {config['pattern']}")

    # Pass 1: Learn statistics
    scaler, label_encoder, lower_bound, upper_bound = pass1_learn_statistics(
        data_dir=config['data_dir'],
        pattern=config['pattern'],
        chunk_size=config['chunk_size']
    )

    # Save scaler and label encoder
    os.makedirs(config['output_dir'], exist_ok=True)

    with open(os.path.join(config['output_dir'], 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(config['output_dir'], 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    with open(os.path.join(config['output_dir'], 'bounds.pkl'), 'wb') as f:
        pickle.dump({'lower': lower_bound, 'upper': upper_bound}, f)

    logger.info(f"\n✓ Saved scaler and label encoder to {config['output_dir']}")

    # Pass 2: Transform and save
    num_chunks = pass2_transform_and_save(
        data_dir=config['data_dir'],
        pattern=config['pattern'],
        chunk_size=config['chunk_size'],
        output_dir=config['output_dir'],
        scaler=scaler,
        label_encoder=label_encoder,
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )

    logger.info("\n" + "="*80)
    logger.info("✓ STEP 1 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutput:")
    logger.info(f"  Directory: {config['output_dir']}")
    logger.info(f"  Chunks: {num_chunks} files (chunk_0000.npz to chunk_{num_chunks-1:04d}.npz)")
    logger.info(f"  Scaler: scaler.pkl")
    logger.info(f"  Label encoder: label_encoder.pkl")
    logger.info(f"  Bounds: bounds.pkl")
    logger.info("\nNext step:")
    logger.info("  python step2_create_federated_splits.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
