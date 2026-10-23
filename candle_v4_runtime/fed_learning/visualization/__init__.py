"""
Visualization module (IEEE Paper Standard)
==========================================

Contains:
1. Standard FL plots (plots.py)
2. Classification metrics (metrics.py) 
3. FCIL-specific visualizations (fcil_plots.py)
"""
from .style import set_ieee_style, get_ieee_figsize, get_ieee_colors
from .plots import plot_training_history, plot_learning_curves, save_training_plots
from .metrics import (
    plot_confusion_matrix, 
    plot_per_class_metrics, 
    plot_roc_curves,
    export_metrics_table
)
from .fcil_plots import (
    # Metrics computation
    compute_average_incremental_accuracy,
    compute_forgetting_measure,
    compute_backward_transfer,
    compute_forward_transfer,
    # Core FCIL plots
    plot_incremental_accuracy_curve,
    plot_forgetting_comparison,
    plot_task_accuracy_heatmap,
    plot_old_new_accuracy,
    plot_bwt_fwt_comparison,
    # Advanced plots
    plot_class_accuracy_evolution,
    plot_convergence_per_task,
    plot_multi_strategy_radar,
    plot_communication_efficiency,
    plot_statistical_comparison,
    # Report generation
    generate_fcil_report,
)

__all__ = [
    # Style
    "set_ieee_style",
    "get_ieee_figsize",
    "get_ieee_colors",
    # Standard FL Plots
    "plot_training_history",
    "plot_learning_curves", 
    "save_training_plots",
    # Classification Metrics
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "plot_roc_curves",
    "export_metrics_table",
    # FCIL Metrics
    "compute_average_incremental_accuracy",
    "compute_forgetting_measure",
    "compute_backward_transfer",
    "compute_forward_transfer",
    # FCIL Plots
    "plot_incremental_accuracy_curve",
    "plot_forgetting_comparison",
    "plot_task_accuracy_heatmap",
    "plot_old_new_accuracy",
    "plot_bwt_fwt_comparison",
    "plot_class_accuracy_evolution",
    "plot_convergence_per_task",
    "plot_multi_strategy_radar",
    "plot_communication_efficiency",
    "plot_statistical_comparison",
    "generate_fcil_report",
]
