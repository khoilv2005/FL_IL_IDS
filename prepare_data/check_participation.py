"""
Check Participation Score (Binary Presence)
===========================================
Tính toán "Điểm tham gia" (Participation Score):
- Nếu Client C có dữ liệu của Class K (> 0 samples) -> Tính là 1 điểm.
- Nếu không có (0 samples) -> 0 điểm.

Script này sẽ tạo ma trận binary [Clients x Classes] và báo cáo thống kê.
"""

import os
import argparse
import numpy as np
import glob
import json
import logging
import pandas as pd
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_class_names(path="class_names.txt"):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f)}
    return {}

def check_participation(data_dir: str):
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found: {data_dir}")
        return

    client_files = sorted(glob.glob(os.path.join(data_dir, "client_*_train.npz")))
    if not client_files:
        logger.error("❌ No client files found!")
        return

    logger.info("=" * 80)
    logger.info(f"📊 ANALYZING PARTICIPATION SCORES (1 Point if Class Present)")
    logger.info(f"   Data Directory: {data_dir}")
    logger.info("=" * 80)

    # 1. Load Metadata to know structure (optional but helpful)
    metadata_path = os.path.join(data_dir, "metadata.json")
    num_classes = 34  # Default
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                num_classes = meta["task_structure"]["total_classes"]
        except:
            pass
    
    # 2. Build Matrix
    # Matrix: rows=clients, cols=classes
    num_clients = len(client_files)
    participation_matrix = np.zeros((num_clients, num_classes), dtype=int)
    
    logger.info(f"Scanning {num_clients} clients...")
    
    for cf in client_files:
        try:
            basename = os.path.basename(cf)
            cid = int(basename.split('_')[1])
            
            data = np.load(cf)
            y = data['y_train']
            
            unique_classes = np.unique(y)
            for cls in unique_classes:
                if cls < num_classes:
                    participation_matrix[cid, cls] = 1
                    
        except Exception as e:
            logger.warning(f"⚠️ Error reading Client {cid}: {e}")

    # 3. Report per Class (Score per Class)
    class_names = load_class_names()
    class_scores = np.sum(participation_matrix, axis=0)
    
    logger.info("\n🏆 SCORE PER CLASS (How many clients hold this class?)")
    logger.info("-" * 80)
    logger.info(f"{'Class ID':^8} | {'Name':<30} | {'Score (Clients)':^15} | {'Diff Imbalance'}")
    logger.info("-" * 80)
    
    sorted_classes = np.argsort(class_scores) # Sort by score
    
    for cls in sorted_classes:
        score = class_scores[cls]
        name = class_names.get(cls, f"Class {cls}")
        bar = "█" * int(score / 2)
        logger.info(f"{cls:^8} | {name:<30} | {score:^15} | {bar}")
        
    # 4. Report per Client (Score per Client)
    client_scores = np.sum(participation_matrix, axis=1)
    
    logger.info("\n🏆 SCORE PER CLIENT (How many classes does client hold?)")
    logger.info(f"   Avg Classes/Client: {np.mean(client_scores):.2f}")
    logger.info(f"   Min: {np.min(client_scores)}, Max: {np.max(client_scores)}")
    
    # Distribution of client scores
    score_dist = pd.Series(client_scores).value_counts().sort_index()
    logger.info("\n   Distribution of Client Scores:")
    for score, count in score_dist.items():
        logger.info(f"     {score} classes: {count} clients")

    # 5. Total Sparsity
    total_entries = num_clients * num_classes
    total_ones = np.sum(participation_matrix)
    sparsity = 1.0 - (total_ones / total_entries)
    
    logger.info("\n📉 OVERALL SPARSITY")
    logger.info(f"   Total Possible Pairs: {total_entries}")
    logger.info(f"   Present Pairs (Ones): {total_ones}")
    logger.info(f"   Sparsity (Zeros)    : {sparsity*100:.2f}% (Higher = More Non-IID)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/federated_splits/100-clients")
    args = parser.parse_args()
    
    check_participation(args.data_dir)
