"""
Quick Check Class 21 - Run this on Kaggle to analyze class 21 distribution
"""

import numpy as np
import os

# Path on Kaggle
data_dir = "/kaggle/input/data-10clients"

print("🔍 Checking Class 21 Distribution")
print("=" * 60)

# Check which task has class 21
# From log: Task 2 has classes [16, 17, 18, 19, 20, 21]
print("\n📚 Class 21 belongs to: Task 2")
print("   Classes in Task 2: [16, 17, 18, 19, 20, 21]")

# Count in each client
total_samples = 0
client_counts = {}

for cid in range(10):  # 10 clients
    try:
        file_path = os.path.join(data_dir, f"client_{cid}_train.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            y = data["y_train"]
            count = np.sum(y == 21)
            if count > 0:
                client_counts[cid] = count
                total_samples += count
                print(f"   Client {cid}: {count} samples")
    except Exception as e:
        print(f"   Client {cid}: Error - {e}")

print(f"\n📊 Total Class 21 samples: {total_samples}")

# Check test data
try:
    test_path = os.path.join(data_dir, "global_test_data.npz")
    if os.path.exists(test_path):
        data = np.load(test_path)
        y_test = data["y_test"]
        test_count = np.sum(y_test == 21)
        print(f"   Test set: {test_count} samples")
except:
    pass

# Check class imbalance
print("\n⚖️ Class Imbalance Analysis:")
if client_counts:
    counts = list(client_counts.values())
    print(
        f"   Max: {max(counts)} samples (Client {max(client_counts, key=client_counts.get)})"
    )
    print(
        f"   Min: {min(counts)} samples (Client {min(client_counts, key=client_counts.get)})"
    )
    print(f"   Avg: {sum(counts) / len(counts):.1f} samples")

    # Check if concentrated in 1-2 clients
    total = sum(counts)
    max_client_ratio = max(counts) / total * 100
    print(f"\n   ⚠️ Class 21 is {max_client_ratio:.1f}% concentrated in 1 client!")

    if max_client_ratio > 50:
        print("   🔴 HIGHLY IMBALANCED - This explains weight explosion!")
        print("   → Model overfits to majority client's class 21 pattern")

print("\n" + "=" * 60)
print("💡 CONCLUSION:")
print("Class 21 is likely the 'last' and 'most represented' class in Task 2.")
print("When μ resets, model amplifies this class's features.")
print("=" * 60)
