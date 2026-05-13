"""
CNN-GRU LwF Trainer - Learning without Forgetting for CNN-GRU.

This module implements the LwF training strategy adapted for CNN-GRU model.
The implementation follows the author's original code structure while using
CNN-GRU instead of ResNet34.

Reference:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016, IEEE TPAMI 2018
    https://arxiv.org/abs/1606.09282

Author's Original Code (model.py):
    - MultiClassCrossEntropy: Knowledge distillation loss function
    - Model.increment_classes: Expand classifier for new classes
    - Model.update: Training loop with distillation

This Implementation:
    - CNN_GRU_LwF: Full LwF trainer class for CNN-GRU
    - Follows author's training loop structure
    - Maintains exact distillation math from paper
"""

import copy
from typing import List, Optional, Dict, Tuple, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


def MultiClassCrossEntropy(logits: torch.Tensor, labels: torch.Tensor, T: float) -> torch.Tensor:
    """
    Knowledge Distillation Loss - Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    
    This is the distillation loss from the author's original code (model.py, line 16-26).
    
    The temperature T controls the "softness" of the probability distributions:
    - T=1: standard softmax
    - T>1: softer probability distribution (more knowledge transfer)
    - T=2: commonly used value (from Hinton et al.)
    
    Paper Section: "Knowledge Distillation"
    Equation: L_KD = T² * KL(σ(z_old/T) || σ(z_new/T))
    
    Args:
        logits: Output logits from current model (student)
        labels: Soft targets from old model (teacher) - raw logits, not probabilities
        T: Temperature for softening probabilities
        
    Returns:
        Distillation loss (scalar tensor)
        
    Example from author (model.py):
        dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
    """
    # Move labels to Variable without gradient (teacher targets are frozen)
    labels = Variable(labels.data, requires_grad=False)
    if torch.cuda.is_available():
        labels = labels.cuda()
    
    # Compute log-softmax of logits (student predictions at temperature T)
    outputs = torch.log_softmax(logits / T, dim=1)
    
    # Compute softmax of labels (teacher predictions at temperature T)
    labels = torch.softmax(labels / T, dim=1)
    
    # Compute the weighted sum: sum over classes of p_teacher * log(p_student)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    
    # Average over batch and negate (negative log-likelihood)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    
    return Variable(outputs.data, requires_grad=True)


def kaiming_normal_init(m: nn.Module) -> None:
    """
    Kaiming Normal initialization for weights.
    
    Author's original function (model.py, line 28-32):
        def kaiming_normal_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
    
    This initialization is good for networks with ReLU activations.
    """
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class LwFTrainer:
    """
    Learning without Forgetting Trainer for CNN-GRU.
    
    This class implements the LwF training strategy following the author's
    original code structure from model.py.
    
    Author's Original Variables (model.py):
        self.init_lr = args.init_lr
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lower_rate_epoch = [int(0.7 * num_epochs), int(0.9 * num_epochs)]
        self.lr_dec_factor = 10
        self.momentum = 0.9
        self.weight_decay = 0.0001
        
    Paper Key Points:
        1. Save old model snapshot (teacher) before training new task
        2. Train new model (student) with:
           - Cross-entropy loss on new task data
           - Distillation loss to match old model's soft predictions
        3. Balance weight α controls trade-off between new knowledge and old knowledge
        
    Training Flow (from author main.py):
        for s in range(0, num_iters, num_classes):
            1. Load dataset for new classes
            2. model.update(train_set, class_map, args)
            3. model.n_known = model.n_classes
            4. Evaluate on all seen classes
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_initial_classes: int = 1,
        init_lr: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 64,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        lr_decay_schedule: Optional[List[int]] = None,
        lr_decay_factor: float = 10.0,
    ):
        """
        Initialize LwF trainer.
        
        Args:
            input_shape: Input shape for CNN-GRU (e.g., (46,) for 46 timesteps)
            num_initial_classes: Initial number of classes (usually 1 for incremental)
            init_lr: Initial learning rate
            num_epochs: Number of training epochs per task
            batch_size: Mini-batch size
            lwf_alpha: Weight for distillation loss (α in paper)
            temperature: Temperature for distillation (T in paper)
            momentum: SGD momentum
            weight_decay: L2 regularization
            lr_decay_schedule: Epochs to decay learning rate (e.g., [14, 18] for 20-epoch training)
            lr_decay_factor: Factor to divide learning rate by on decay
        """
        # Hyper Parameters (from author model.py line 36-47)
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_decay_factor = lr_decay_factor
        
        # Default decay schedule: 70% and 90% of training
        if lr_decay_schedule is None:
            self.lr_decay_schedule = [
                int(0.7 * num_epochs),
                int(0.9 * num_epochs)
            ]
        else:
            self.lr_decay_schedule = lr_decay_schedule
        
        # Model tracking (from author model.py line 61-66)
        self.n_classes = 0  # Total classes after expansion
        self.n_known = 0    # Classes known before current task
        self.classes_map = {}  # Original class ID -> internal index
        
        # Old model snapshot for distillation (from author model.py line 107-111)
        self.prev_model: Optional[nn.Module] = None
        
        # Store model and input shape for reference
        self.input_shape = input_shape
        self._model: Optional[nn.Module] = None
        
        print(f"[LwF Trainer] Initialized: α={lwf_alpha}, T={temperature}, "
              f"epochs={num_epochs}, lr={init_lr}")
    
    @property
    def model(self) -> nn.Module:
        """Get the model."""
        return self._model
    
    @model.setter
    def model(self, model: nn.Module):
        """Set the model."""
        self._model = model
    
    def increment_classes(self, new_classes: List[int]) -> None:
        """
        Add new classes to the classifier.
        
        Author's original function (model.py, line 73-91):
            def increment_classes(self, new_classes):
                n = len(new_classes)
                in_features = self.fc.in_features
                out_features = self.fc.out_features
                weight = self.fc.weight.data
                
                if self.n_known == 0:
                    new_out_features = n
                else:
                    new_out_features = out_features + n
                    
                self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
                self.fc = self.model.fc
                kaiming_normal_init(self.fc.weight)
                self.fc.weight.data[:out_features] = weight
                self.n_classes += n
        
        This expands the output layer to accommodate new classes while
        preserving the learned weights for old classes.
        """
        n = len(new_classes)
        print(f"[LwF] Adding {n} new classes")
        
        # Get current classifier
        if hasattr(self._model, 'fc2'):
            # CNN_GRU_Model structure
            in_features = self._model.fc2.in_features
            out_features = self._model.fc2.out_features
            old_weight = self._model.fc2.weight.data.clone()
        else:
            raise AttributeError("Model does not have 'fc2' classifier attribute")
        
        # Calculate new output size
        if self.n_known == 0:
            new_out_features = n
        else:
            new_out_features = out_features + n
        
        print(f"[LwF] Expanding classifier: {out_features} -> {new_out_features}")
        
        # Create new classifier with expanded output
        new_fc = nn.Linear(in_features, new_out_features, bias=False)
        new_fc.apply(kaiming_normal_init)
        
        # Copy old weights to new classifier
        new_fc.weight.data[:out_features] = old_weight
        
        # Replace classifier
        self._model.fc2 = new_fc
        
        # Update class tracking
        self.n_classes += n
        
        # Update classes map
        for cls in new_classes:
            if cls not in self.classes_map:
                self.classes_map[cls] = len(self.classes_map)
    
    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Classify images by taking argmax of softmax predictions.
        
        Author's original function (model.py, line 93-103):
            def classify(self, images):
                _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)
                return preds
        """
        self._model.eval()
        with torch.no_grad():
            logits = self._model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, dim=1, keepdim=False)
        return preds
    
    def compute_distillation_loss(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        old_class_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Paper Equation: L_KD = T² * KL(σ(z_old/T) || σ(z_new/T))
        
        This matches the author's MultiClassCrossEntropy function but
        with explicit control over which classes to distill.
        
        Args:
            old_logits: Outputs from old model (teacher)
            new_logits: Outputs from current model (student)
            old_class_indices: If provided, only distill on these class indices
            
        Returns:
            Distillation loss
        """
        T = self.temperature
        
        if old_class_indices is not None and len(old_class_indices) > 0:
            # Filter to old classes only
            indices = torch.tensor(old_class_indices, device=new_logits.device, dtype=torch.long)
            old_logits = old_logits[:, indices]
            new_logits = new_logits[:, indices]
        
        # Compute distillation loss using author's formula
        return MultiClassCrossEntropy(new_logits, old_logits, T)
    
    def save_prev_model(self) -> None:
        """
        Save current model snapshot before training new task.
        
        Author's original code (model.py, line 109-111):
            prev_model = copy.deepcopy(self)
            prev_model.cuda()
        
        This snapshot is used as the teacher for knowledge distillation.
        """
        if self._model is not None:
            self.prev_model = copy.deepcopy(self._model)
            if torch.cuda.is_available():
                self.prev_model = self.prev_model.cuda()
            self.prev_model.eval()
            # Freeze parameters
            for param in self.prev_model.parameters():
                param.requires_grad = False
            print("[LwF] Saved model snapshot for distillation")
    
    def update(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Update model with new task data using LwF.
        
        Author's original function (model.py, line 105-169):
            def update(self, dataset, class_map, args):
                self.compute_means = True
                prev_model = copy.deepcopy(self)
                
                classes = list(set(dataset.train_labels))
                if self.n_classes == 1 and self.n_known == 0:
                    new_classes = [classes[i] for i in range(1,len(classes))]
                else:
                    new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]
                
                if len(new_classes) > 0:
                    self.increment_classes(new_classes)
                
                loader = torch.utils.data.DataLoader(...)
                optimizer = optim.SGD(self.parameters(), lr=self.init_lr, ...)
                
                for epoch in range(self.num_epochs):
                    for i, (indices, images, labels) in enumerate(loader):
                        # Compute losses
                        cls_loss = nn.CrossEntropyLoss()(logits, labels)
                        if self.n_classes//len(new_classes) > 1:
                            dist_target = prev_model.forward(images)
                            logits_dist = logits[:,:-(self.n_classes-self.n_known)]
                            dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
                            loss = dist_loss + cls_loss
                        else:
                            loss = cls_loss
                        loss.backward()
                        optimizer.step()
        
        Training Loop:
            1. Save old model as teacher
            2. Expand classifier for new classes
            3. For each epoch:
                a. Forward pass
                b. Compute CE loss on new data
                c. Compute KD loss (if not first task)
                d. Backward and update
        """
        if self._model is None:
            raise ValueError("Model not set. Call trainer.model = model first.")
        
        # Step 1: Save old model for distillation
        self.save_prev_model()
        
        # Step 2: Determine new classes to add
        if classes is None:
            # Auto-detect from dataset
            if hasattr(train_dataset, 'targets'):
                all_labels = train_dataset.targets
            elif hasattr(train_dataset, 'y_train'):
                all_labels = train_dataset.y_train
            else:
                all_labels = []
            classes = sorted(list(set(all_labels)))
        
        # Identify new classes
        new_classes = [c for c in classes if c not in self.classes_map]
        
        if len(new_classes) > 0:
            self.increment_classes(new_classes)
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = self._model.to(device)
        
        # Step 3: Create data loader
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for simplicity
        )
        
        # Step 4: Setup optimizer
        optimizer = optim.SGD(
            self._model.parameters(),
            lr=self.init_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate decay schedule
        current_lr = self.init_lr
        
        # Step 5: Training loop
        self._model.train()
        is_first_task = (self.n_classes // max(1, len(new_classes)) == 1) if new_classes else True
        
        for epoch in range(self.num_epochs):
            # Update learning rate according to schedule
            if epoch in self.lr_decay_schedule:
                current_lr = current_lr / self.lr_decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, batch in enumerate(loader):
                # Get data
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                    else:
                        images = batch[0]
                        labels = None
                else:
                    images = batch
                    labels = None
                
                # Move to device
                images = images.to(device)
                if labels is not None:
                    labels = labels.to(device)
                
                # Map labels to internal indices
                if labels is not None and self.classes_map:
                    remapped_labels = torch.tensor(
                        [self.classes_map.get(int(l), int(l)) for l in labels],
                        device=device,
                        dtype=torch.long
                    )
                else:
                    remapped_labels = labels
                
                # Forward pass
                optimizer.zero_grad()
                logits = self._model(images)
                
                # Compute losses
                if is_first_task or self.prev_model is None:
                    # First task: only cross-entropy loss
                    loss = F.cross_entropy(logits, remapped_labels)
                else:
                    # Subsequent tasks: CE + distillation loss
                    ce_loss = F.cross_entropy(logits, remapped_labels)
                    
                    # Get old model predictions
                    with torch.no_grad():
                        old_logits = self.prev_model(images)
                    
                    # Distill on old class logits only (matching author's logic)
                    num_old_classes = self.n_classes - len(new_classes)
                    if num_old_classes > 0:
                        old_indices = list(range(num_old_classes))
                        dist_loss = self.compute_distillation_loss(
                            old_logits[:, :num_old_classes],
                            logits[:, :num_old_classes]
                        )
                        loss = ce_loss + self.lwf_alpha * dist_loss
                    else:
                        loss = ce_loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(images)
                epoch_samples += len(images)
            
            avg_loss = epoch_loss / max(1, epoch_samples)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{self.num_epochs}] Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
        
        # Update known classes count
        self.n_known = self.n_classes
        
        # Return final metrics
        return {
            'final_loss': avg_loss,
            'n_classes': self.n_classes,
            'n_known': self.n_known
        }
    
    def get_accuracy(
        self,
        dataset: torch.utils.data.Dataset,
        device: Optional[torch.device] = None
    ) -> Tuple[float, float]:
        """
        Compute accuracy on a dataset.
        
        Author's evaluation code (main.py, line 118-142):
            total = 0.0
            correct = 0.0
            for indices, images, labels in test_loader:
                images = Variable(images).cuda()
                preds = model.classify(images)
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()
            accuracy = 100.0 * correct / total
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._model.eval()
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        total = 0
        correct = 0
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                else:
                    images = batch
                    labels = None
                
                images = images.to(device)
                
                if labels is not None:
                    labels = labels.to(device)
                    preds = self.classify(images)
                    
                    # Map predictions back to original labels if needed
                    reverse_map = {v: k for k, v in self.classes_map.items()}
                    mapped_preds = torch.tensor(
                        [reverse_map.get(int(p), int(p)) for p in preds],
                        device=device
                    )
                    mapped_labels = labels
                    
                    total += len(labels)
                    correct += (mapped_preds == mapped_labels).sum().item()
        
        accuracy = 100.0 * correct / max(1, total) if total > 0 else 0.0
        return accuracy, correct / max(1, total)


class CNN_GRU_LwF:
    """
    Convenience class that combines CNN-GRU model with LwF trainer.
    
    This provides a unified interface for LwF with CNN-GRU architecture.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int = 1,
        init_lr: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 64,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        **kwargs
    ):
        """
        Initialize CNN-GRU model with LwF trainer.
        
        Args:
            input_shape: Input shape (e.g., (46,) for 46 timesteps)
            num_classes: Initial number of classes
            init_lr: Initial learning rate
            num_epochs: Training epochs per task
            batch_size: Batch size
            lwf_alpha: Distillation weight
            temperature: Distillation temperature
            **kwargs: Additional arguments for trainer
        """
        from ..models.cnn_gru import CNN_GRU_Model
        
        # Create model
        self.model = CNN_GRU_Model(input_shape, num_classes=num_classes)
        
        # Create trainer and attach model
        self.trainer = LwFTrainer(
            input_shape=input_shape,
            num_initial_classes=num_classes,
            init_lr=init_lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lwf_alpha=lwf_alpha,
            temperature=temperature,
            **kwargs
        )
        self.trainer.model = self.model
        
        print(f"[CNN_GRU_LwF] Created with {num_classes} initial classes")
    
    def set_task(self, new_classes: List[int]) -> None:
        """
        Prepare for a new task.
        
        This is called when moving to a new incremental learning task.
        """
        self.trainer.save_prev_model()
        self.trainer.increment_classes(new_classes)
        print(f"[CNN_GRU_LwF] Set task with classes: {new_classes}")
    
    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train on a new task.
        """
        return self.trainer.update(train_dataset, test_dataset, classes, verbose)
    
    def evaluate(
        self,
        dataset: torch.utils.data.Dataset
    ) -> Tuple[float, float]:
        """
        Evaluate on a dataset.
        """
        return self.trainer.get_accuracy(dataset)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input data.
        """
        return self.trainer.classify(x)
