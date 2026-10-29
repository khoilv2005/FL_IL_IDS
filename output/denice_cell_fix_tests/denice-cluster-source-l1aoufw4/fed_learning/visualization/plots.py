"""
Training Visualization Plots (IEEE Standard)
=============================================
"""

import os
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np

from .style import (
    set_ieee_style, 
    get_ieee_figsize, 
    get_ieee_colors,
    get_ieee_markers,
    get_ieee_linestyles
)


def plot_training_history(history: Dict, 
                          figsize: tuple = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history with loss and metrics curves (IEEE style).
    
    Args:
        history: Training history dictionary with keys like 'train_loss', 'test_accuracy', etc.
        figsize: Figure size (width, height). Defaults to IEEE double column.
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('double', aspect=0.6)
    
    fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[0] * 0.7))
    fig.suptitle("Federated Learning Training History", fontsize=10, fontweight='bold')
    
    rounds = range(1, len(history.get("train_loss", [])) + 1)
    colors = get_ieee_colors(4)
    markers = get_ieee_markers(4)
    
    # Plot 1: Loss
    ax1 = axes[0, 0]
    if "train_loss" in history and history["train_loss"]:
        ax1.plot(rounds, history["train_loss"], color=colors[0], marker=markers[0], 
                 linestyle='-', label='Train Loss', markersize=4, markevery=max(1, len(rounds)//10))
    if "test_loss" in history and history["test_loss"]:
        eval_rounds = range(1, len(history["test_loss"]) + 1)
        ax1.plot(eval_rounds, history["test_loss"], color=colors[1], marker=markers[1],
                 linestyle='--', label='Test Loss', markersize=4, markevery=max(1, len(eval_rounds)//10))
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Training & Test Loss")
    ax1.legend(loc='upper right')
    
    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    if "test_accuracy" in history and history["test_accuracy"]:
        eval_rounds = range(1, len(history["test_accuracy"]) + 1)
        acc_percent = [a * 100 for a in history["test_accuracy"]]
        ax2.plot(eval_rounds, acc_percent, color=colors[2], marker=markers[2],
                 linestyle='-', label='Test Accuracy', markersize=4, markevery=max(1, len(eval_rounds)//10))
        ax2.set_ylim([0, 100])
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b) Test Accuracy")
    ax2.legend(loc='lower right')
    
    # Plot 3: F1 Scores
    ax3 = axes[1, 0]
    if "test_f1_macro" in history and history["test_f1_macro"]:
        eval_rounds = range(1, len(history["test_f1_macro"]) + 1)
        f1_macro = [f * 100 for f in history["test_f1_macro"]]
        f1_weighted = [f * 100 for f in history.get("test_f1_weighted", history["test_f1_macro"])]
        ax3.plot(eval_rounds, f1_macro, color=colors[0], marker=markers[0],
                 linestyle='-', label='F1 (Macro)', markersize=4, markevery=max(1, len(eval_rounds)//10))
        ax3.plot(eval_rounds, f1_weighted, color=colors[1], marker=markers[1],
                 linestyle='--', label='F1 (Weighted)', markersize=4, markevery=max(1, len(eval_rounds)//10))
        ax3.set_ylim([0, 100])
    ax3.set_xlabel("Communication Round")
    ax3.set_ylabel("F1 Score (%)")
    ax3.set_title("(c) F1 Scores")
    ax3.legend(loc='lower right')
    
    # Plot 4: Precision & Recall
    ax4 = axes[1, 1]
    if "test_precision_macro" in history and history["test_precision_macro"]:
        eval_rounds = range(1, len(history["test_precision_macro"]) + 1)
        precision = [p * 100 for p in history["test_precision_macro"]]
        recall = [r * 100 for r in history.get("test_recall_macro", history["test_precision_macro"])]
        ax4.plot(eval_rounds, precision, color=colors[2], marker=markers[2],
                 linestyle='-', label='Precision (Macro)', markersize=4, markevery=max(1, len(eval_rounds)//10))
        ax4.plot(eval_rounds, recall, color=colors[3], marker=markers[3],
                 linestyle='--', label='Recall (Macro)', markersize=4, markevery=max(1, len(eval_rounds)//10))
        ax4.set_ylim([0, 100])
    ax4.set_xlabel("Communication Round")
    ax4.set_ylabel("Score (%)")
    ax4.set_title("(d) Precision & Recall")
    ax4.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        # Save as both PNG and PDF
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"📊 Saved: {save_path} & {pdf_path}")
    
    return fig


def plot_learning_curves(history: Dict,
                         figsize: tuple = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning curves - train vs test loss (IEEE style).
    
    Args:
        history: Training history dictionary
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('single')
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = get_ieee_colors(2)
    markers = get_ieee_markers(2)
    
    if "train_loss" in history and history["train_loss"]:
        rounds = range(1, len(history["train_loss"]) + 1)
        ax.plot(rounds, history["train_loss"], color=colors[0], marker=markers[0],
                linestyle='-', label='Train Loss', markersize=4, markevery=max(1, len(rounds)//8))
    
    if "test_loss" in history and history["test_loss"]:
        eval_rounds = range(1, len(history["test_loss"]) + 1)
        ax.plot(eval_rounds, history["test_loss"], color=colors[1], marker=markers[1],
                linestyle='--', label='Test Loss', markersize=4, markevery=max(1, len(eval_rounds)//8))
    
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend(loc='upper right')
    
    # Annotate minimum test loss
    if "test_loss" in history and history["test_loss"]:
        min_idx = np.argmin(history["test_loss"])
        min_val = history["test_loss"][min_idx]
        ax.annotate(f'Min: {min_val:.4f}', 
                    xy=(min_idx + 1, min_val),
                    xytext=(min_idx + 1.5, min_val + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                    fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"📊 Saved: {save_path} & {pdf_path}")
    
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
