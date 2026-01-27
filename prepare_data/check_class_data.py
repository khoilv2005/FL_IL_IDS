"""
Check Class Data Distribution
=============================
Kiểm tra xem một class cụ thể được phân phối như thế nào qua các clients.

Usage:
    python check_class_data.py --class_id 10
    python check_class_data.py --class_id 10 --data_dir ./data/federated_splits/100-clients
"""

import os
import argparse
import numpy as np
import glob
import json
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_class_names(path="class_names.txt"):
    """Load class names map if available."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f)}
    return {}

def check_class_distribution(data_dir, target_class_id):
    """Scan all client files and report distribution of target_class_id."""
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found: {data_dir}")
        return

    # Load metadata if exists to get context
    metadata_path = os.path.join(data_dir, "metadata.json")
    task_info = ""
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                task_struct = meta.get("task_structure", {})
                
                # Find which task this class belongs to
                found_task = -1
                for task_id, classes in task_struct.get("task_classes", {}).items():
                    if target_class_id in classes:
                        found_task = task_id
                        break
                if found_task != -1:
                    task_info = f"(Belongs to Task {found_task})"
        except:
            pass

    class_names = load_class_names()
    class_name = class_names.get(target_class_id, f"Class {target_class_id}")
    
    logger.info("=" * 80)
    logger.info(f"🧐 INSPECTING CLASS {target_class_id}: {class_name} {task_info}")
    logger.info(f"   Data Directory: {data_dir}")
    logger.info("=" * 80)

    client_files = sorted(glob.glob(os.path.join(data_dir, "client_*_train.npz")))
    
    if not client_files:
        logger.error("❌ No client data files found!")
        return

    client_counts = {}
    total_samples = 0
    total_clients_holding = 0
    
    # 1. Scan all files
    logger.info(f"Scanning {len(client_files)} client files...")
    
    for cf in client_files:
        try:
            # Extract client ID from filename "client_123_train.npz"
            basename = os.path.basename(cf)
            cid = int(basename.split('_')[1])
            
            data = np.load(cf)
            y = data['y_train']
            
            # Count samples for target class
            count = np.sum(y == target_class_id)
            
            if count > 0:
                client_counts[cid] = count
                total_samples += count
                total_clients_holding += 1
                
        except Exception as e:
            logger.warning(f"⚠️ Error reading {cf}: {e}")

    # 2. Add Test Data info
    test_path = os.path.join(data_dir, "global_test_data.npz")
    test_count = 0
    if os.path.exists(test_path):
        try:
            data = np.load(test_path)
            y_test = data['y_test']
            test_count = np.sum(y_test == target_class_id)
        except:
            pass

    # 3. Print Report
    if total_samples == 0:
        logger.warning(f"\n❌ Class {target_class_id} not found in any client training data!")
        if test_count > 0:
             logger.info(f"   However, found {test_count:,} samples in Global Test Set.")
        return

    logger.info("\n📊 DISTRIBUTION REPORT")
    logger.info(f"   Total Train Samples: {total_samples:,}")
    logger.info(f"   Held by Clients: {total_clients_holding} / {len(client_files)}")
    if test_count > 0:
        logger.info(f"   Global Test Samples: {test_count:,}")

    logger.info("\n📋 DETAILS BY CLIENT")
    logger.info("-" * 80)
    logger.info(f"{'Client ID':^10} | {'Samples':^12} | {'% of Class Data':^20} | {'Bar Chart':<30}")
    logger.info("-" * 80)

    # Sort by sample count descending
    sorted_clients = sorted(client_counts.items(), key=lambda x: x[1], reverse=True)

    for cid, count in sorted_clients:
        percent = (count / total_samples) * 100
        bar_len = int(percent / 2)  # 50 chars = 100%
        bar = "█" * bar_len
        
        logger.info(f"{cid:^10} | {count:^12,} | {percent:^19.2f}% | {bar:<30}")

    logger.info("-" * 80)
    
    # 4. Summary Statistics
    counts = list(client_counts.values())
    logger.info("\n📈 STATISTICS")
    logger.info(f"   Min samples/client: {min(counts):,}")
    logger.info(f"   Max samples/client: {max(counts):,}")
    logger.info(f"   Avg samples/client: {np.mean(counts):.1f}")
    logger.info(f"   Std samples/client: {np.std(counts):.1f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check class data distribution across clients")
    parser.add_argument("--class_id", type=int, required=True, help="ID of the class to inspect")
    parser.add_argument("--data_dir", type=str, default="./data/federated_splits/100-clients", 
                        help="Directory containing federated data splits")
    
    args = parser.parse_args()
    
    check_class_distribution(args.data_dir, args.class_id)
