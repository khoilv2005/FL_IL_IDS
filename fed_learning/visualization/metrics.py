"""
Metrics Visualization - Confusion Matrix, ROC Curves, Per-class Metrics
"""

import os
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: bool = True,
                          figsize: tuple = (12, 10),
                          cmap: str = 'Blues',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        normalize: If True, normalize by row (true labels)
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with 0
    
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved confusion matrix: {save_path}")
    
    return fig


def plot_per_class_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           figsize: tuple = (14, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1 scores as bar chart.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    n_classes = len(precision)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    x = np.arange(n_classes)
    width = 0.25
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Precision, Recall, F1
    ax1 = axes[0]
    bars1 = ax1.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax1.bar(x, recall, width, label='Recall', color='orange')
    bars3 = ax1.bar(x + width, f1, width, label='F1-Score', color='green')
    
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Support (sample count per class)
    ax2 = axes[1]
    bars4 = ax2.bar(x, support, color='purple', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Support (Samples per Class)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved per-class metrics: {save_path}")
    
    return fig


def plot_roc_curves(y_true: np.ndarray,
                    y_proba: np.ndarray,
                    num_classes: int,
                    class_names: Optional[List[str]] = None,
                    figsize: tuple = (10, 8),
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels (1D array of class indices)
        y_proba: Predicted probabilities (shape: n_samples x n_classes)
        num_classes: Number of classes
        class_names: Optional list of class names
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute ROC curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves (limit to first 10 classes for readability)
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_classes, 10)))
    for i in range(min(num_classes, 10)):
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved ROC curves: {save_path}")
    
    return fig
