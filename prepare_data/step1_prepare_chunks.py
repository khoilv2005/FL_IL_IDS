"""
STEP 1 v3: Chunked Preprocessing (NO SCALING, NO BOUNDS)
=========================================================
Script này xử lý 45M samples với RAM hạn chế:
1. Đọc từng file CSV
2. Encode labels
3. Save RAW data to disk (NO clipping, NO scaling)

QUAN TRỌNG: 
- KHÔNG fit StandardScaler ở đây!
- KHÔNG tính outlier bounds ở đây!
- Scaler và bounds sẽ được tính trong step2 CHỈ trên TRAIN data
- Tránh 100% data leak từ test → train
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import glob
import logging
import pickle
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': './data/IoT_Dataset_2023',
    'output_dir': './data/raw_chunks',
    'chunk_size': 100000,
    'pattern': 'Merged*.csv',
}

# ============================================================================
# PASS 1: Collect labels ONLY
# ============================================================================

def pass1_collect_labels(data_dir, pattern, chunk_size):
    """Collect all unique labels (no leak - labels are same across split)"""
    logger.info("=" * 80)
    logger.info("PASS 1: COLLECTING LABELS ONLY")
    logger.info("=" * 80)

    csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    logger.info(f"Found {len(csv_files)} CSV files")

    label_encoder = LabelEncoder()
    all_labels = []

    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{len(csv_files)}] Scanning: {os.path.basename(csv_file)}")
        
        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_df.columns = chunk_df.columns.str.lower()
            if 'label' in chunk_df.columns:
                chunk_df = chunk_df[chunk_df['label'].notna()]
                all_labels.extend(chunk_df['label'].unique())

    unique_labels = list(set(all_labels))
    label_encoder.fit(unique_labels)
    
    logger.info(f"✓ Found {len(unique_labels)} unique labels")
    logger.info(f"  Labels: {label_encoder.classes_}")
    logger.info("\n⚠️  Outlier bounds will be computed in step2 (TRAIN-only)")

    return label_encoder

# ============================================================================
# PASS 2: Save RAW chunks
# ============================================================================

def pass2_save_raw_chunks(data_dir, pattern, chunk_size, output_dir, label_encoder):
    """Save RAW data (NO scale, NO clip) - processing happens in step2"""
    logger.info("\n" + "=" * 80)
    logger.info("PASS 2: SAVING RAW CHUNKS")
    logger.info("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    chunk_id = 0
    total_size = 0

    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"\n[{i}/{len(csv_files)}] Processing: {os.path.basename(csv_file)}")

        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_df.columns = chunk_df.columns.str.lower()

            if 'label' not in chunk_df.columns:
                continue

            chunk_df = chunk_df[chunk_df['label'].notna()]
            if len(chunk_df) == 0:
                continue

            X_chunk = chunk_df.drop(['label'], axis=1).values
            y_chunk = chunk_df['label'].values

            # Handle NaN/inf ONLY (no scaling, no clipping)
            X_chunk = np.nan_to_num(X_chunk, nan=0.0, posinf=1e10, neginf=-1e10)
            X_chunk = X_chunk.astype(np.float32)

            # Encode labels
            y_chunk = label_encoder.transform(y_chunk)

            # Save chunk
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}.npz")
            np.savez_compressed(chunk_path, X=X_chunk, y=y_chunk)

            file_size = os.path.getsize(chunk_path) / (1024 ** 2)
            total_size += file_size
            logger.info(f"  Chunk {chunk_id:04d}: {len(X_chunk):,} samples, {file_size:.2f} MB")
            chunk_id += 1

    logger.info(f"\n✓ Saved {chunk_id} RAW chunks ({total_size:.2f} MB)")
    return chunk_id

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("STEP 1 v3: CHUNKED PREPROCESSING (ZERO LEAK)")
    logger.info("=" * 80)
    logger.info("\n⚠️  This script saves RAW data only")
    logger.info("   Scaling and clipping will be done in step2 on TRAIN-only")

    config = CONFIG

    # Pass 1: Collect labels
    label_encoder = pass1_collect_labels(
        data_dir=config['data_dir'],
        pattern=config['pattern'],
        chunk_size=config['chunk_size']
    )

    # Save label encoder
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"\n✓ Saved label encoder to {config['output_dir']}")

    # Pass 2: Save RAW chunks
    num_chunks = pass2_save_raw_chunks(
        data_dir=config['data_dir'],
        pattern=config['pattern'],
        chunk_size=config['chunk_size'],
        output_dir=config['output_dir'],
        label_encoder=label_encoder
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ STEP 1 COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput: {config['output_dir']}")
    logger.info(f"  - {num_chunks} RAW chunks")
    logger.info(f"  - label_encoder.pkl")
    logger.info("\n⚠️  IMPORTANT: These are RAW, unprocessed!")
    logger.info("   Run step2_federated_splits.py to scale (TRAIN-only)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
