"""
LwF Main Training Script for CNN-GRU.

This script implements the incremental learning training loop following
the author's original main.py structure.

Author's Original Structure (main.py):
    1. Parse arguments
    2. Setup data (CIFAR-100 with class permutation)
    3. Create model
    4. Loop through tasks:
        a. Load task data
        b. Update model (train with LwF)
        c. Update n_known
        d. Evaluate on train/test
        e. Save accuracy matrix

This Implementation:
    - Uses CNN-GRU instead of ResNet34
    - Uses incremental_loader.py for IDS data
    - Follows exact training loop structure
    - Produces same output format as author

Reference:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016, IEEE TPAMI 2018
    https://arxiv.org/abs/1606.09282
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fed_learning.methods.lwf.lwf_trainer import CNN_GRU_LwF, LwFTrainer
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.utils.seed import set_seed


def parse_args():
    """Parse command line arguments - matching author's main.py."""
    parser = argparse.ArgumentParser(description='LwF Incremental Learning with CNN-GRU')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing federated data files')
    parser.add_argument('--output_dir', type=str, default='./lwf_results',
                       help='Directory for output files')
    
    # Model arguments
    parser.add_argument('--input_shape', type=int, nargs='+', default=[46],
                       help='Input shape (e.g., 46 for 46 timesteps)')
    parser.add_argument('--num_classes', type=int, default=34,
                       help='Total number of classes in dataset')
    
    # Incremental learning arguments
    parser.add_argument('--num_tasks', type=int, default=6,
                       help='Number of incremental tasks')
    parser.add_argument('--classes_per_task', type=int, default=6,
                       help='Number of classes per task')
    
    # Training arguments (matching author)
    parser.add_argument('--init_lr', type=float, default=0.1,
                       help='Initial learning rate (author default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=40,
                       help='Number of epochs per task (author default: 40)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Mini-batch size (author default: 64)')
    
    # LwF specific arguments
    parser.add_argument('--lwf_alpha', type=float, default=1.0,
                       help='Weight for distillation loss (α in paper)')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature for distillation (author default: 2.0)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (author default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='L2 weight decay (author default: 0.0001)')
    
    # Output arguments
    parser.add_argument('--outfile', type=str, default='lwf_results.csv',
                       help='Output CSV file for accuracy results')
    parser.add_argument('--save_model', action='store_true',
                       help='Save model after each task')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    return args


class LwFTrainingLoop:
    """
    Main training loop following author's structure from main.py.
    
    Author's Original Loop (main.py line 87-170):
        for s in range(0, num_iters, num_classes):
            # Load Datasets
            print('Iteration: ', s)
            train_set = cifar100(...)
            train_loader = torch.utils.data.DataLoader(train_set, ...)
            test_set = cifar100(...)
            test_loader = torch.utils.data.DataLoader(test_set, ...)
            
            # Update representation via BackProp
            model.update(train_set, class_map, args)
            model.eval()
            
            model.n_known = model.n_classes
            
            # Train Accuracy
            ...
            
            # Test Accuracy
            ...
            
            # Accuracy matrix
            ...
            
            model.train()
    
    This implementation follows the same structure with adaptations for CNN-GRU
    and IDS data.
    """
    
    def __init__(self, args):
        """Initialize training loop with arguments."""
        self.args = args
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_seed(args.seed)
        
        # Load data loader
        self.data_loader = IncrementalDataLoader(args.data_dir)
        
        # Parse input shape
        if len(args.input_shape) == 1:
            self.input_shape = tuple(args.input_shape)
        else:
            self.input_shape = tuple(args.input_shape)
        
        # Get task structure from data
        self.num_tasks = args.num_tasks
        self.classes_per_task = args.classes_per_task
        self.total_classes = args.num_classes
        
        # Calculate classes per task from data if available
        if hasattr(self.data_loader, 'get_num_tasks'):
            self.num_tasks = self.data_loader.get_num_tasks()
            task_classes = self.data_loader.get_task_classes(0)
            self.classes_per_task = len(task_classes)
        
        # Class mapping
        self.class_map = {}  # original_class_id -> internal_index
        self.map_reverse = {}  # internal_index -> original_class_id
        
        # History tracking
        self.history = {
            'train_accuracy': [],
            'test_accuracy': [],
            'task_accuracies': [],  # Per-class accuracy
            'train_loss': [],
        }
        
        # Accuracy matrix (per task, per iteration)
        self.acc_matr = np.zeros((self.num_tasks, self.num_tasks))
        
        print(f"Input shape: {self.input_shape}")
        print(f"Total classes: {self.total_classes}")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Classes per task: {self.classes_per_task}")
    
    def _build_class_mapping(self, all_classes: List[int]) -> None:
        """
        Build class mapping from original IDs to internal indices.
        
        Author's code (main.py line 52-66):
            n_cl_temp = 0
            for i, cl in enumerate(all_classes):
                if cl not in class_map:
                    class_map[cl] = int(n_cl_temp)
                    n_cl_temp += 1
            
            for cl, map_cl in class_map.items():
                map_reverse[map_cl] = int(cl)
        """
        n_cl_temp = 0
        for cl in all_classes:
            if cl not in self.class_map:
                self.class_map[cl] = int(n_cl_temp)
                n_cl_temp += 1
        
        # Build reverse mapping
        self.map_reverse = {v: k for k, v in self.class_map.items()}
    
    def create_model(self) -> CNN_GRU_Model:
        """
        Create CNN-GRU model for LwF.
        
        Uses the CNN_GRU_Model from fed_learning/models/cnn_gru.py
        instead of ResNet34 from author's code.
        """
        model = CNN_GRU_Model(self.input_shape, num_classes=self.classes_per_task)
        model = model.to(self.device)
        return model
    
    def get_task_dataset(
        self,
        task_id: int,
        client_id: int = 0
    ) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
        """
        Get dataset for a specific task.
        
        Uses incremental_loader.py to load task data.
        """
        # Get training data for this client and task
        X_train, y_train = self.data_loader.get_client_data(client_id, task_id)
        
        if len(y_train) == 0:
            # Try another client or aggregate
            client_ids = self.data_loader.get_all_client_ids()
            for cid in client_ids:
                X_train, y_train = self.data_loader.get_client_data(cid, task_id)
                if len(y_train) > 0:
                    break
        
        # Get test data (cumulative: all seen classes so far)
        X_test, y_test = self.data_loader.get_test_data(task_id, cumulative=True)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        
        return train_dataset, test_dataset
    
    def train_task(
        self,
        model: nn.Module,
        trainer: LwFTrainer,
        task_id: int,
        train_dataset: torch.utils.data.TensorDataset,
        test_dataset: Optional[torch.utils.data.TensorDataset] = None
    ) -> Dict[str, float]:
        """
        Train on a single task using LwF.
        
        This function follows the author's update() structure.
        """
        print(f"\n{'='*60}")
        print(f"Task {task_id}: Training with LwF")
        print(f"{'='*60}")
        
        # Get classes for this task
        task_classes = self.data_loader.get_task_classes(task_id)
        internal_classes = [self.class_map.get(c, c) for c in task_classes]
        
        # Set task in trainer
        trainer.set_task(task_id, internal_classes)
        
        # Save previous model for distillation
        trainer.save_prev_model()
        
        # Increment classes in model
        if task_id > 0:
            trainer.increment_classes(internal_classes)
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer (following author)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.args.init_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        # Training loop (matching author structure)
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        current_lr = self.args.init_lr
        lr_decay_schedule = [
            int(0.7 * self.args.num_epochs),
            int(0.9 * self.args.num_epochs)
        ]
        
        model.train()
        is_first_task = (task_id == 0)
        
        for epoch in range(self.args.num_epochs):
            # Learning rate decay (author's schedule)
            if epoch in lr_decay_schedule:
                current_lr = current_lr / 10.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Map labels to internal indices
                remapped_labels = torch.tensor(
                    [self.class_map.get(int(l), int(l)) for l in labels.cpu()],
                    device=self.device,
                    dtype=torch.long
                )
                
                optimizer.zero_grad()
                logits = model(images)
                
                # Compute losses
                if is_first_task:
                    loss = nn.CrossEntropyLoss()(logits, remapped_labels)
                else:
                    ce_loss = nn.CrossEntropyLoss()(logits, remapped_labels)
                    
                    # Distillation loss
                    with torch.no_grad():
                        old_logits = trainer.prev_model(images)
                    
                    num_old_classes = trainer.n_known
                    if num_old_classes > 0:
                        dist_loss = trainer.compute_distillation_loss(
                            old_logits[:, :num_old_classes],
                            logits[:, :num_old_classes]
                        )
                        loss = ce_loss + self.args.lwf_alpha * dist_loss
                    else:
                        loss = ce_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(images)
                epoch_samples += len(images)
            
            avg_loss = epoch_loss / max(1, epoch_samples)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{self.args.num_epochs}] Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
        
        # Update trainer state
        trainer.n_known = trainer.n_classes
        model.eval()
        
        return {'final_loss': avg_loss, 'n_classes': trainer.n_classes}
    
    def evaluate(
        self,
        model: nn.Module,
        dataset: torch.utils.data.TensorDataset,
        prefix: str = "Test"
    ) -> Tuple[float, float]:
        """
        Evaluate model on dataset.
        
        Author's evaluation code (main.py line 118-142):
            total = 0.0
            correct = 0.0
            for indices, images, labels in test_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
            accuracy = 100.0 * correct / total
        """
        model.eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        
        total = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                logits = model(images)
                _, preds = torch.max(logits, dim=1)
                
                # Map predictions back to original labels
                mapped_preds = torch.tensor(
                    [self.map_reverse.get(int(p), int(p)) for p in preds.cpu()],
                    device=self.device
                )
                
                total += len(labels)
                correct += (mapped_preds == labels).sum().item()
        
        accuracy = 100.0 * correct / max(1, total)
        return accuracy, correct / max(1, total)
    
    def evaluate_per_class(
        self,
        model: nn.Module,
        task_id: int
    ) -> np.ndarray:
        """
        Evaluate accuracy per class (author's accuracy matrix).
        
        Author's code (main.py line 144-164):
            for i in range(model.n_known):
                test_set = cifar100(classes=all_classes[i*num_classes: (i+1)*num_classes])
                ...
                acc_matr[i, int(s/num_classes)] = (100 * correct / total)
        """
        accuracies = np.zeros(self.num_tasks)
        
        for t in range(task_id + 1):
            # Get test data for this specific task's classes
            X_test, y_test = self.data_loader.get_test_data(t, cumulative=False)
            
            if len(y_test) == 0:
                continue
            
            dataset = torch.utils.data.TensorDataset(X_test, y_test)
            acc, _ = self.evaluate(model, dataset, prefix=f"Task {t}")
            accuracies[t] = acc
        
        return accuracies
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete incremental learning loop.
        
        Author's main loop (main.py line 87-170):
            for s in range(0, num_iters, num_classes):
                # Load data
                # Train
                # Evaluate
                # Save results
        """
        print(f"\n{'='*60}")
        print("LwF Incremental Learning with CNN-GRU")
        print(f"{'='*60}")
        
        # Get all classes from data
        all_classes = []
        for t in range(self.num_tasks):
            task_cls = self.data_loader.get_task_classes(t)
            all_classes.extend(task_cls)
        all_classes = sorted(list(set(all_classes)))
        
        # Build class mapping
        self._build_class_mapping(all_classes)
        
        print(f"Class map: {self.class_map}")
        print(f"Map reverse: {self.map_reverse}")
        
        # Create model and trainer
        model = self.create_model()
        trainer = LwFTrainer(
            input_shape=self.input_shape,
            num_initial_classes=self.classes_per_task,
            init_lr=self.args.init_lr,
            num_epochs=self.args.num_epochs,
            batch_size=self.args.batch_size,
            lwf_alpha=self.args.lwf_alpha,
            temperature=self.args.temperature,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        trainer.model = model
        
        # Track seen classes
        seen_classes = []
        
        # Training loop
        start_time = time.time()
        
        for task_id in range(self.num_tasks):
            # Get task classes
            task_classes = self.data_loader.get_task_classes(task_id)
            print(f"\n>>> Task {task_id}: classes {task_classes}")
            
            # Update seen classes
            seen_classes.extend(task_classes)
            
            # Get dataset
            train_dataset, test_dataset = self.get_task_dataset(task_id)
            print(f"    Train samples: {len(train_dataset)}")
            print(f"    Test samples (cumulative): {len(test_dataset)}")
            
            # Train
            train_results = self.train_task(model, trainer, task_id, train_dataset, test_dataset)
            
            # Update seen classes in trainer
            trainer.seen_classes = seen_classes.copy()
            
            # Evaluate on cumulative test set
            cumulative_test_dataset = torch.utils.data.TensorDataset(*self.data_loader.get_test_data(task_id, cumulative=True))
            test_acc, _ = self.evaluate(model, cumulative_test_dataset, prefix="Cumulative Test")
            
            print(f"\n>>> Task {task_id} Results:")
            print(f"    Test Accuracy (cumulative): {test_acc:.2f}%")
            
            # Evaluate per class
            per_class_acc = self.evaluate_per_class(model, task_id)
            self.acc_matr[task_id, :task_id+1] = per_class_acc[:task_id+1]
            
            print(f"    Per-task accuracy: {per_class_acc}")
            
            # Save history
            self.history['test_accuracy'].append(test_acc)
            self.history['train_loss'].append(train_results['final_loss'])
            self.history['task_accuracies'].append(per_class_acc.tolist())
            
            # Save model if requested
            if self.args.save_model:
                model_path = self.output_dir / f"model_task_{task_id}.pt"
                torch.save({
                    'task_id': task_id,
                    'model_state': model.state_dict(),
                    'trainer_state': {
                        'n_classes': trainer.n_classes,
                        'n_known': trainer.n_known,
                        'classes_map': trainer.classes_map,
                    }
                }, model_path)
                print(f"    Model saved to {model_path}")
            
            # Print accuracy matrix
            print(f"\n>>> Accuracy Matrix (rows=seen tasks, cols=iterations):")
            print(self.acc_matr[:task_id+1, :task_id+1])
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final test accuracy: {self.history['test_accuracy'][-1]:.2f}%")
        
        # Average accuracy across all tasks
        avg_acc = np.mean([self.history['task_accuracies'][t][t] for t in range(self.num_tasks)])
        print(f"Average per-task accuracy: {avg_acc:.2f}%")
        
        # Average forgetting
        if self.num_tasks > 1:
            forgetting = 0.0
            count = 0
            for t1 in range(self.num_tasks):
                for t2 in range(t1 + 1, self.num_tasks):
                    if t1 < len(self.history['task_accuracies'][t2]):
                        drop = self.history['task_accuracies'][t1][t1] - self.history['task_accuracies'][t2][t1]
                        forgetting += max(0, drop)
                        count += 1
            avg_forgetting = forgetting / max(1, count) if count > 0 else 0.0
            print(f"Average Forgetting: {avg_forgetting:.2f}%")
        
        # Save results
        self._save_results()
        
        return {
            'history': self.history,
            'accuracy_matrix': self.acc_matr,
            'total_time': total_time
        }
    
    def _save_results(self) -> None:
        """Save training results to files."""
        # CSV results
        csv_path = self.output_dir / self.args.outfile
        with open(csv_path, 'w') as f:
            f.write("Task,Train_Accuracy,Test_Accuracy\n")
            for i, (train_acc, test_acc) in enumerate(zip(
                self.history.get('train_accuracy', [0]*len(self.history['test_accuracy'])),
                self.history['test_accuracy']
            )):
                f.write(f"{i},{train_acc:.2f},{test_acc:.2f}\n")
        
        # Accuracy matrix
        np.save(self.output_dir / 'accuracy_matrix.npy', self.acc_matr)
        
        # Full history as JSON
        history_path = self.output_dir / 'training_history.json'
        # Convert numpy types to Python types for JSON serialization
        history_json = {
            'test_accuracy': [float(x) for x in self.history['test_accuracy']],
            'train_loss': [float(x) for x in self.history['train_loss']],
            'task_accuracies': [
                [float(x) for x in ta] for ta in self.history['task_accuracies']
            ]
        }
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("="*60)
    print("LwF (Learning without Forgetting) for CNN-GRU")
    print("="*60)
    print(f"Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()
    
    # Run training
    loop = LwFTrainingLoop(args)
    results = loop.run()
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
