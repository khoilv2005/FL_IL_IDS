"""
Main Training Loop for Federated Learning
"""

from typing import Dict

from tqdm.auto import tqdm


def train_federated_multigpu(server, config: Dict) -> Dict:
    """
    Main training loop for federated learning.
    
    Args:
        server: FederatedServerMultiGPU instance
        config: Training configuration
    
    Returns:
        Training history dictionary
    """
    R = config["num_rounds"]
    eval_every = config["eval_every"]
    history = server.history
    
    for ridx in tqdm(range(R), desc="Global Rounds"):
        print(f"\n{'='*60}")
        print(f"ROUND {ridx+1}/{R}")
        print(f"{'='*60}")
        
        # Train
        r_res = server.train_round(verbose=True)
        
        # Always record train loss
        history["train_loss"].append(r_res["train_loss"])
        
        # Evaluate
        if (ridx + 1) % eval_every == 0:
            print("\n  📊 Evaluating global model...")
            metrics = server.evaluate_global()
            
            history["test_loss"].append(metrics["loss"])
            history["test_accuracy"].append(metrics["accuracy"])
            history["test_f1_macro"].append(metrics["f1_macro"])
            history["test_f1_weighted"].append(metrics["f1_weighted"])
            history["test_precision_macro"].append(metrics["precision_macro"])
            history["test_recall_macro"].append(metrics["recall_macro"])
            history["test_auc_macro"].append(metrics.get("auc_macro_ovr"))
            
            print(f"\n{'='*60}")
            print(f"📊 METRICS SUMMARY - Round {ridx+1}")
            print(f"{'='*60}")
            print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
            print(f"  F1 (macro):         {metrics['f1_macro']*100:.2f}%")
            print(f"  F1 (weighted):      {metrics['f1_weighted']*100:.2f}%")
            print(f"  Precision (macro):  {metrics['precision_macro']*100:.2f}%")
            print(f"  Recall (macro):     {metrics['recall_macro']*100:.2f}%")
            if metrics.get("auc_macro_ovr"):
                print(f"  AUC (macro OvR):    {metrics['auc_macro_ovr']*100:.2f}%")
            print(f"{'='*60}")
    
    return history
