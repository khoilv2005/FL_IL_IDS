"""
Training Visualization Plots
"""

import os
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history: Dict, 
                          figsize: tuple = (14, 10),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history with loss and metrics curves.
    
    Args:
        history: Training history dictionary with keys like 'train_loss', 'test_accuracy', etc.
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Federated Learning Training History", fontsize=14, fontweight='bold')
    
    rounds = range(1, len(history.get("train_loss", [])) + 1)
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    if "train_loss" in history and history["train_loss"]:
        ax1.plot(rounds, history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    if "test_loss" in history and history["test_loss"]:
        eval_rounds = range(1, len(history["test_loss"]) + 1)
        ax1.plot(eval_rounds, history["test_loss"], 'r-s', label='Test Loss', linewidth=2, markersize=4)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    if "test_accuracy" in history and history["test_accuracy"]:
        eval_rounds = range(1, len(history["test_accuracy"]) + 1)
        acc_percent = [a * 100 for a in history["test_accuracy"]]
        ax2.plot(eval_rounds, acc_percent, 'g-o', label='Test Accuracy', linewidth=2, markersize=4)
        ax2.set_ylim([0, 100])
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F1 Scores
    ax3 = axes[1, 0]
    if "test_f1_macro" in history and history["test_f1_macro"]:
        eval_rounds = range(1, len(history["test_f1_macro"]) + 1)
        f1_macro = [f * 100 for f in history["test_f1_macro"]]
        f1_weighted = [f * 100 for f in history.get("test_f1_weighted", history["test_f1_macro"])]
        ax3.plot(eval_rounds, f1_macro, 'purple', marker='o', label='F1 Macro', linewidth=2, markersize=4)
        ax3.plot(eval_rounds, f1_weighted, 'orange', marker='s', label='F1 Weighted', linewidth=2, markersize=4)
        ax3.set_ylim([0, 100])
    ax3.set_xlabel("Round")
    ax3.set_ylabel("F1 Score (%)")
    ax3.set_title("F1 Scores")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision & Recall
    ax4 = axes[1, 1]
    if "test_precision_macro" in history and history["test_precision_macro"]:
        eval_rounds = range(1, len(history["test_precision_macro"]) + 1)
        precision = [p * 100 for p in history["test_precision_macro"]]
        recall = [r * 100 for r in history.get("test_recall_macro", history["test_precision_macro"])]
        ax4.plot(eval_rounds, precision, 'cyan', marker='o', label='Precision (Macro)', linewidth=2, markersize=4)
        ax4.plot(eval_rounds, recall, 'magenta', marker='s', label='Recall (Macro)', linewidth=2, markersize=4)
        ax4.set_ylim([0, 100])
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Score (%)")
    ax4.set_title("Precision & Recall (Macro)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved training history plot: {save_path}")
    
    return fig


def plot_learning_curves(history: Dict,
                         figsize: tuple = (10, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot simple learning curves (train vs test loss).
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if "train_loss" in history and history["train_loss"]:
        rounds = range(1, len(history["train_loss"]) + 1)
        ax.plot(rounds, history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    
    if "test_loss" in history and history["test_loss"]:
        eval_rounds = range(1, len(history["test_loss"]) + 1)
        ax.plot(eval_rounds, history["test_loss"], 'r-s', label='Test Loss', linewidth=2, markersize=6)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Learning Curves", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for min loss
    if "test_loss" in history and history["test_loss"]:
        min_idx = np.argmin(history["test_loss"])
        min_val = history["test_loss"][min_idx]
        ax.annotate(f'Min: {min_val:.4f}', 
                    xy=(min_idx + 1, min_val),
                    xytext=(min_idx + 1, min_val + 0.1),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved learning curves: {save_path}")
    
    return fig


def save_training_plots(history: Dict, output_dir: str, prefix: str = ""):
    """
    Save all training plots to output directory.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
        prefix: Optional prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Full history plot
    history_path = os.path.join(output_dir, f"{prefix}training_history.png")
    fig1 = plot_training_history(history, save_path=history_path)
    plt.close(fig1)
    
    # Learning curves
    curves_path = os.path.join(output_dir, f"{prefix}learning_curves.png")
    fig2 = plot_learning_curves(history, save_path=curves_path)
    plt.close(fig2)
    
    print(f"✅ All training plots saved to: {output_dir}")
