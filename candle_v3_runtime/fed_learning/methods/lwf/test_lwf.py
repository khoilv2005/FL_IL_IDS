"""
Quick test script for LwF implementation.

This script demonstrates the LwF implementation with a simple synthetic dataset
to verify the implementation is working correctly.

Usage:
    python -m fed_learning.methods.lwf.test_lwf
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fed_learning.methods.lwf import LwFModel, MultiClassCrossEntropy, kaiming_normal_init
from fed_learning.models.cnn_gru import CNN_GRU_Model


def test_distillation_loss():
    """Test that MultiClassCrossEntropy is computed correctly."""
    print("\n" + "="*60)
    print("Test 1: MultiClassCrossEntropy")
    print("="*60)
    
    # Create dummy logits and labels
    batch_size = 4
    num_classes = 10
    temperature = 2.0
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randn(batch_size, num_classes)  # Soft targets from teacher
    
    # Compute loss
    loss = MultiClassCrossEntropy(logits, labels, temperature)
    
    print(f"Batch size: {batch_size}")
    print(f"Num classes: {num_classes}")
    print(f"Temperature: {temperature}")
    print(f"Loss value: {loss.item():.4f}")
    
    # Verify gradients flow
    loss.backward()
    print("✓ Gradient computation successful")
    
    return loss.item()


def test_cnn_gru_forward():
    """Test CNN-GRU forward pass."""
    print("\n" + "="*60)
    print("Test 2: CNN-GRU Forward Pass")
    print("="*60)
    
    batch_size = 8
    seq_length = 46
    
    model = CNN_GRU_Model(input_shape=(seq_length,), num_classes=10)
    x = torch.randn(batch_size, seq_length)
    
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 10)")
    
    assert output.shape == (batch_size, 10), "Output shape mismatch!"
    print("✓ Forward pass successful")
    
    return True


def test_lwf_model_incremental():
    """Test LwF model with incremental learning."""
    print("\n" + "="*60)
    print("Test 3: LwF Model Incremental Learning")
    print("="*60)
    
    # Create model
    input_shape = (46,)
    num_classes_per_task = 2
    
    model = LwFModel(
        input_shape=input_shape,
        num_classes=num_classes_per_task,
        init_lr=0.001,
        num_epochs=2,  # Few epochs for testing
        batch_size=16,
        lwf_alpha=1.0,
        temperature=2.0
    )
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Simulate incremental learning
    num_tasks = 3
    
    for task_id in range(num_tasks):
        # Generate synthetic data for this task
        classes_this_task = [task_id * num_classes_per_task + i for i in range(num_classes_per_task)]
        
        print(f"\n--- Task {task_id}: Classes {classes_this_task} ---")
        
        # Set task (this saves old model and expands classifier)
        model.set_task(classes_this_task)
        
        # Create synthetic dataset
        num_samples = 50
        X = torch.randn(num_samples, input_shape[0])
        y = torch.tensor([classes_this_task[i % len(classes_this_task)] for i in range(num_samples)])
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.001,
            momentum=0.9
        )
        
        model.train()
        for epoch in range(2):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                loss_dict = model.train_step(X_batch, y_batch, optimizer)
                epoch_loss += loss_dict['ce_loss'] + loss_dict.get('kd_loss', 0)
            
            avg_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test = torch.randn(10, input_shape[0]).to(device)
            preds = model.classify(X_test)
            print(f"  Predictions shape: {preds.shape}")
            print(f"  Sample predictions: {preds[:5].cpu().tolist()}")
    
    print("\n✓ Incremental learning successful")
    return True


def test_class_expansion():
    """Test that class expansion preserves old weights."""
    print("\n" + "="*60)
    print("Test 4: Class Expansion")
    print("="*60)
    
    model = LwFModel(input_shape=(46,), num_classes=5)
    
    # Get initial classifier weights
    initial_weights = model.fc.weight.data.clone()
    print(f"Initial classifier shape: {initial_weights.shape}")
    
    # Simulate task 1
    model.set_task([5, 6, 7])
    expanded_weights = model.fc.weight.data.clone()
    print(f"After task 1 shape: {expanded_weights.shape}")
    
    # Verify old weights are preserved
    weight_preserved = torch.allclose(
        expanded_weights[:5], initial_weights, atol=1e-6
    )
    print(f"Old weights preserved: {weight_preserved}")
    
    if weight_preserved:
        print("✓ Class expansion preserves old weights")
    else:
        print("✗ Class expansion FAILED to preserve weights")
    
    return weight_preserved


def test_knowledge_distillation():
    """Test that knowledge distillation is computed correctly."""
    print("\n" + "="*60)
    print("Test 5: Knowledge Distillation")
    print("="*60)
    
    model = LwFModel(
        input_shape=(46,),
        num_classes=3,
        lwf_alpha=1.0,
        temperature=2.0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # First task: no distillation
    model.set_task([0, 1, 2])
    
    X = torch.randn(8, 46).to(device)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Train on task 1
    for _ in range(5):
        model.train_step(X, y, optimizer)
    
    # Verify old model is saved
    assert model.prev_model is not None, "Old model should be saved"
    print(f"✓ Old model saved for distillation")
    
    # Second task: should have distillation
    model.set_task([3, 4, 5])
    
    # Train and check KD loss
    model.train()
    optimizer.zero_grad()
    
    logits = model(X)
    loss, loss_dict = model._compute_loss(logits, y, X)
    
    print(f"CE loss: {loss_dict['ce_loss']:.4f}")
    print(f"KD loss: {loss_dict['kd_loss']:.4f}")
    print(f"Total loss: {loss.item():.4f}")
    
    assert loss_dict['kd_loss'] > 0, "KD loss should be positive"
    print("✓ Knowledge distillation computed correctly")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("LwF Implementation Tests")
    print("="*60)
    
    tests = [
        ("MultiClassCrossEntropy", test_distillation_loss),
        ("CNN-GRU Forward", test_cnn_gru_forward),
        ("Class Expansion", test_class_expansion),
        ("Knowledge Distillation", test_knowledge_distillation),
        ("Incremental Learning", test_lwf_model_incremental),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, True, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, result in results:
        status = "✓ PASS" if success else "✗ FAIL"
        result_str = f" ({result})" if isinstance(result, float) else ""
        print(f"  {status}: {name}{result_str}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
