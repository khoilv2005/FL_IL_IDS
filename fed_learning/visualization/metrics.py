"""
Metrics Visualization - Confusion Matrix, ROC Curves, Per-class Metrics (IEEE Standard)
========================================================================================
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

from .style import (
    set_ieee_style,
    get_ieee_figsize,
    get_ieee_colors,
    get_ieee_linestyles
)


def plot_confusion_matrix(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: bool = True,
                          figsize: tuple = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix heatmap (IEEE style).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        normalize: If True, normalize by row (true labels)
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    set_ieee_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    if figsize is None:
        # Scale figure based on number of classes
        size = min(10, max(5, n_classes * 0.3))
        figsize = (size, size)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use grayscale colormap for IEEE
    im = ax.imshow(cm, interpolation='nearest', cmap='Greys')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    
    # Show all ticks
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    
    # Add text annotations (only if classes < 20 for readability)
    if n_classes <= 20:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=6)
    
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"📊 Saved: {save_path} & {pdf_path}")
    
    return fig


def plot_per_class_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           figsize: tuple = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1 scores as bar chart (IEEE style).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure object
    """
    set_ieee_style()
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    n_classes = len(precision)
    
    if class_names is None:
        class_names = [f"C{i}" for i in range(n_classes)]
    
    if figsize is None:
        figsize = get_ieee_figsize('double', aspect=0.4)
    
    x = np.arange(n_classes)
    width = 0.25
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = get_ieee_colors(4)
    
    # Plot 1: Precision, Recall, F1
    ax1 = axes[0]
    ax1.bar(x - width, precision, width, label='Precision', color=colors[0], edgecolor='black', linewidth=0.5)
    ax1.bar(x, recall, width, label='Recall', color=colors[1], edgecolor='black', linewidth=0.5)
    ax1.bar(x + width, f1, width, label='F1-Score', color=colors[2], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Score')
    ax1.set_title('(a) Per-Class Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax1.legend(fontsize=7, loc='lower right')
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Support (sample count per class)
    ax2 = axes[1]
    bars = ax2.bar(x, support, color=colors[3], edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('(b) Support (Samples per Class)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"📊 Saved: {save_path} & {pdf_path}")
    
    return fig


def plot_roc_curves(y_true: np.ndarray,
                    y_proba: np.ndarray,
                    num_classes: int,
                    class_names: Optional[List[str]] = None,
                    figsize: tuple = None,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification (IEEE style).
    
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
    set_ieee_style()
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    if figsize is None:
        figsize = get_ieee_figsize('single', aspect=1.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute ROC curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Get colors and line styles for IEEE
    max_classes_to_plot = min(num_classes, 10)
    colors = get_ieee_colors(max_classes_to_plot)
    linestyles = get_ieee_linestyles(max_classes_to_plot)
    
    for i in range(max_classes_to_plot):
        ax.plot(fpr[i], tpr[i], color=colors[i % len(colors)], 
                linestyle=linestyles[i % len(linestyles)], lw=1.2,
                label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc="lower right", fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"📊 Saved: {save_path} & {pdf_path}")
    
    return fig


def export_metrics_table(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         output_path: Optional[str] = None) -> str:
    """
    Export per-class metrics as LaTeX table for IEEE papers.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        output_path: Optional path to save the LaTeX table
    
    Returns:
        LaTeX table string
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    n_classes = len(precision)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Build LaTeX table
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Per-Class Classification Metrics}")
    latex.append(r"\label{tab:metrics}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\")
    latex.append(r"\hline")
    
    for i in range(n_classes):
        latex.append(f"{class_names[i]} & {precision[i]:.3f} & {recall[i]:.3f} & {f1[i]:.3f} & {int(support[i])} \\\\")
    
    latex.append(r"\hline")
    
    # Add macro and weighted averages
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)
    total_support = np.sum(support)
    
    weighted_p = np.average(precision, weights=support)
    weighted_r = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    latex.append(f"Macro Avg & {macro_p:.3f} & {macro_r:.3f} & {macro_f1:.3f} & {int(total_support)} \\\\")
    latex.append(f"Weighted Avg & {weighted_p:.3f} & {weighted_r:.3f} & {weighted_f1:.3f} & {int(total_support)} \\\\")
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"📊 Saved LaTeX table: {output_path}")
    
    return latex_str
