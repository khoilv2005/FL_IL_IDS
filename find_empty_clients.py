"""
Script to find which clients don't have data for specific classes.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fed_learning.data.incremental_loader import IncrementalDataLoader

def find_all_empty_clients(data_dir: str):
    """Find clients that don't have data for ANY class across all tasks."""
    print(f"Loading data from: {data_dir}")
    loader = IncrementalDataLoader(data_dir)

    # Track data per client per class
    client_classes = defaultdict(set)  # client_id -> set of classes
    clients_with_no_data = []

    for task_id in range(6):  # 6 tasks total
        for client_id in range(100):
            X_train, y_train = loader.get_client_data(client_id, task_id)

            if len(y_train) == 0:
                continue

            y_np = y_train.numpy() if hasattr(y_train, 'numpy') else y_train
            for cls in y_np:
                client_classes[client_id].add(int(cls))

    # Find clients with no data at all
    for client_id in range(100):
        if client_id not in client_classes or len(client_classes[client_id]) == 0:
            clients_with_no_data.append(client_id)

    # Summary
    print(f"\n{'='*60}")
    print(f"Total clients: 100")
    print(f"Clients with NO data at all: {len(clients_with_no_data)}")
    if clients_with_no_data:
        print(f"Client IDs: {clients_with_no_data}")

    # Show class distribution
    print(f"\nClients with data by class count:")
    class_counts = defaultdict(list)
    for cid, classes in client_classes.items():
        class_counts[len(classes)].append(cid)

    for count in sorted(class_counts.keys()):
        print(f"  {len(class_counts[count])} clients have {count} classes")

    return clients_with_no_data, client_classes

def find_clients_without_task(data_dir: str, task_id: int, target_classes: list):
    """Find clients that don't have data for specific classes in a given task."""
    print(f"\n{'='*60}")
    print(f"Task {task_id}: classes {target_classes}")
    print(f"{'='*60}")

    loader = IncrementalDataLoader(data_dir)

    clients_without_data = []
    clients_with_data = []

    for client_id in range(100):
        X_train, y_train = loader.get_client_data(client_id, task_id)

        if len(y_train) == 0:
            clients_without_data.append({'client_id': client_id, 'reason': 'empty'})
            continue

        y_np = y_train.numpy() if hasattr(y_train, 'numpy') else y_train

        has_target = any(cls in y_np for cls in target_classes)

        if has_target:
            clients_with_data.append(client_id)
        else:
            unique_classes = sorted(set(y_np))
            clients_without_data.append({
                'client_id': client_id,
                'num_samples': len(y_np),
                'classes': unique_classes
            })

    print(f"Clients WITH data: {len(clients_with_data)}")
    print(f"Clients WITHOUT data: {len(clients_without_data)}")

    for c in clients_without_data:
        if c['reason'] == 'empty':
            print(f"  Client {c['client_id']}: EMPTY")
        else:
            print(f"  Client {c['client_id']}: {c['num_samples']} samples, classes: {c['classes']}")

    return clients_without_data, clients_with_data

if __name__ == "__main__":
    data_dir = "D:/Project/FL_IL_IDS/data/federated_splits/100-clients"

    # Task definitions: 6 classes per task
    # Task 0: 0-5, Task 1: 6-11, Task 2: 12-17, Task 3: 18-23, Task 4: 24-29, Task 5: 30-33
    task_classes = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29],
        [30, 31, 32, 33]
    ]

    # Check all tasks
    all_empty = []
    for task_id, classes in enumerate(task_classes):
        empty, _ = find_clients_without_task(data_dir, task_id, classes)
        all_empty.extend([c['client_id'] for c in empty if 'reason' in c and c['reason'] == 'empty'])

    print(f"\n{'='*60}")
    print(f"SUMMARY: Clients with NO data in ANY task: {len(set(all_empty))}")
    print(f"Client IDs: {sorted(set(all_empty))}")
