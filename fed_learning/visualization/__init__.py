"""Visualization module (IEEE Paper Standard)"""
from .style import set_ieee_style, get_ieee_figsize, get_ieee_colors
from .plots import plot_training_history, plot_learning_curves, save_training_plots
from .metrics import (
    plot_confusion_matrix, 
    plot_per_class_metrics, 
    plot_roc_curves,
    export_metrics_table
)

__all__ = [
    # Style
    "set_ieee_style",
    "get_ieee_figsize",
    "get_ieee_colors",
    # Plots
    "plot_training_history",
    "plot_learning_curves", 
    "save_training_plots",
    # Metrics
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "plot_roc_curves",
    "export_metrics_table",
]
