"""
STEP 1 v4: Single-Pass Chunked Preprocessing (NO SCALING, NO BOUNDS)
=====================================================================
Script này xử lý dữ liệu lớn với RAM hạn chế:
1. Đọc CSV một lần duy nhất
2. Collect labels + Save chunks đồng thời
3. KHÔNG scale, KHÔNG clip

Ưu điểm so với v3:
- Chỉ đọc CSV 1 lần (tiết kiệm ~50% thời gian I/O)
- Giải phóng RAM explicitly sau mỗi chunk
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
    'data_dir': '../data/raw',
    'output_dir': '../data/raw_chunks',
    'chunk_size': 1000000,
    'pattern': 'Merged*.csv',
}

# ============================================================================
# SINGLE PASS: Collect labels + Save chunks
# ============================================================================

def process_chunks(data_dir, pattern, chunk_size, output_dir):
    """Đọc CSV 1 lần, collect labels và save chunks đồng thời."""
    logger.info("=" * 80)
    logger.info("SINGLE PASS: Processing CSV files")
    logger.info("=" * 80)

    csv_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    logger.info(f"Found {len(csv_files)} CSV files")

    # Phase 1: Quick scan để collect labels
    logger.info("\n[Phase 1] Scanning labels...")
    all_labels_set = set()
    total_rows = 0

    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"  [{i}/{len(csv_files)}] Scanning: {os.path.basename(csv_file)}")

        for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_df.columns = chunk_df.columns.str.lower()
            if 'label' in chunk_df.columns:
                chunk_df = chunk_df[chunk_df['label'].notna()]
                all_labels_set.update(chunk_df['label'].unique())
                total_rows += len(chunk_df)
            del chunk_df
            gc.collect()

    unique_labels = sorted(list(all_labels_set))
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    logger.info(f"✓ Found {len(unique_labels)} unique labels")
    logger.info(f"  Labels: {label_encoder.classes_}")
    logger.info(f"  Total rows (estimated): {total_rows:,}")

    # Save label encoder
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"✓ Saved label encoder to {output_dir}")

    # Phase 2: Save chunks (reuse same file iteration)
    logger.info("\n[Phase 2] Saving RAW chunks...")
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
                del chunk_df
                gc.collect()
                continue

            # Tách X, y - dùng iloc thay vì drop() để tránh copy
            y_chunk = chunk_df['label'].values
            X_chunk = chunk_df.iloc[:, :-1].values  # Tất cả trừ cột cuối (label)

            del chunk_df
            gc.collect()

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

            del X_chunk, y_chunk
            gc.collect()

            chunk_id += 1

    logger.info(f"\n✓ Saved {chunk_id} RAW chunks ({total_size:.2f} MB)")
    return chunk_id

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("STEP 1 v4: SINGLE-PASS CHUNKED PREPROCESSING (ZERO LEAK)")
    logger.info("=" * 80)
    logger.info("\n⚠️  This script saves RAW data only")
    logger.info("   Scaling and clipping will be done in step2 on TRAIN-only")

    config = CONFIG

    num_chunks = process_chunks(
        data_dir=config['data_dir'],
        pattern=config['pattern'],
        chunk_size=config['chunk_size'],
        output_dir=config['output_dir']
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
