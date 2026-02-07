"""
FCIL Visualization - Federated Class-Incremental Learning (IEEE/A* Conference Standard)
========================================================================================

Standard plots for FCIL papers:
1. Incremental Accuracy Curve (ACC_task vs Task)
2. Forgetting Measure Bar Chart
3. Backward/Forward Transfer Matrix
4. Task-wise Accuracy Heatmap
5. Class Accuracy Evolution (Old vs New)
6. Multi-strategy Comparison Plots
7. Convergence Analysis per Task
8. Memory/Communication Efficiency

Reference papers:
- CGoFed (CVPR 2024)
- FedCBDR (ECCV 2024)  
- GLFC (CVPR 2022)
- TARGET (ICLR 2023)
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

from .style import (
    set_ieee_style,
    get_ieee_figsize,
    get_ieee_colors,
    get_ieee_markers,
    get_ieee_linestyles
)


# ==============================================================================
# FCIL METRICS COMPUTATION
# ==============================================================================

def compute_average_incremental_accuracy(task_accuracies: Dict[int, List[float]]) -> List[float]:
    """
    Compute Average Incremental Accuracy (AIA) after each task.
    
    AIA_t = (1/t) * sum_{i=1}^{t} A_{t,i}
    
    where A_{t,i} is accuracy on task i after learning task t.
    
    Args:
        task_accuracies: {task_id: [acc_on_task_0, acc_on_task_1, ..., acc_on_task_t]}
    
    Returns:
        List of AIA values for each task
    """
    aia_values = []
    for t in sorted(task_accuracies.keys()):
        accs = task_accuracies[t][:t+1]  # Only accuracies on tasks 0..t
        aia = np.mean(accs)
        aia_values.append(aia)
    return aia_values


def compute_forgetting_measure(task_accuracies: Dict[int, List[float]]) -> List[float]:
    """
    Compute Forgetting Measure for each old task.
    
    F_j = max_{t ∈ {j,...,T-1}} (A_{t,j}) - A_{T,j}
    
    Forgetting = average over all old tasks.
    
    Args:
        task_accuracies: {task_id: [acc_on_task_0, ..., acc_on_task_t]}
    
    Returns:
        List of forgetting values for each old task (task 0 to T-2)
    """
    T = max(task_accuracies.keys())
    forgetting = []
    
    for j in range(T):  # Old tasks 0 to T-1
        max_acc = 0
        for t in range(j, T + 1):
            if t in task_accuracies and j < len(task_accuracies[t]):
                max_acc = max(max_acc, task_accuracies[t][j])
        
        final_acc = task_accuracies[T][j] if j < len(task_accuracies[T]) else 0
        forgetting.append(max_acc - final_acc)
    
    return forgetting


def compute_backward_transfer(task_accuracies: Dict[int, List[float]]) -> float:
    """
    Compute Backward Transfer (BWT).
    
    BWT = (1/(T-1)) * sum_{i=1}^{T-1} (A_{T,i} - A_{i,i})
    
    Negative BWT indicates forgetting.
    
    Args:
        task_accuracies: {task_id: [acc_on_task_0, ..., acc_on_task_t]}
    
    Returns:
        BWT value (negative = forgetting, positive = improvement)
    """
    T = max(task_accuracies.keys())
    if T == 0:
        return 0.0
    
    bwt_sum = 0
    count = 0
    
    for i in range(T):
        if i < len(task_accuracies[T]) and i < len(task_accuracies[i]):
            final_acc = task_accuracies[T][i]
            initial_acc = task_accuracies[i][i]
            bwt_sum += (final_acc - initial_acc)
            count += 1
    
    return bwt_sum / count if count > 0 else 0.0


def compute_forward_transfer(task_accuracies: Dict[int, List[float]], 
                             random_baseline: float = 0.0) -> float:
    """
    Compute Forward Transfer (FWT).
    
    FWT = (1/(T-1)) * sum_{i=2}^{T} (A_{i-1,i} - baseline)
    
    Positive FWT indicates knowledge transfer to new tasks.
    
    Args:
        task_accuracies: {task_id: [acc_on_task_0, ..., acc_on_task_t]}
        random_baseline: Random baseline accuracy
    
    Returns:
        FWT value
    """
    T = max(task_accuracies.keys())
    if T == 0:
        return 0.0
    
    fwt_sum = 0
    count = 0
    
    for i in range(1, T + 1):
        if i in task_accuracies and i-1 in task_accuracies:
            # Accuracy on task i right before learning task i
            if i < len(task_accuracies.get(i-1, [])):
                acc_before = task_accuracies[i-1][i] if i < len(task_accuracies[i-1]) else 0
                fwt_sum += (acc_before - random_baseline)
                count += 1
    
    return fwt_sum / count if count > 0 else 0.0


# ==============================================================================
# CORE FCIL PLOTS
# ==============================================================================

def plot_incremental_accuracy_curve(
    results: Dict[str, Dict[int, List[float]]],
    figsize: tuple = None,
    save_path: Optional[str] = None,
    title: str = None,
    show_std: bool = True,
    metric: str = 'aia'  # 'aia' or 'final'
) -> plt.Figure:
    """
    Plot Average Incremental Accuracy (AIA) curve across tasks.
    
    This is THE standard plot for FCIL papers showing how accuracy 
    evolves as new tasks are learned.
    
    Args:
        results: {strategy_name: {task_id: [acc_task_0, ..., acc_task_t]}}
                 Can also include std: {strategy_name + '_std': {...}}
        figsize: Figure size
        save_path: Optional path to save
        title: Optional custom title
        show_std: Whether to show standard deviation bands
        metric: 'aia' for Average Incremental Accuracy, 'final' for final task accuracy
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('single', aspect=0.8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out std entries
    strategies = [k for k in results.keys() if not k.endswith('_std')]
    colors = get_ieee_colors(len(strategies))
    markers = get_ieee_markers(len(strategies))
    linestyles = get_ieee_linestyles(len(strategies))
    
    for idx, strategy in enumerate(strategies):
        task_accs = results[strategy]
        
        if metric == 'aia':
            y_values = compute_average_incremental_accuracy(task_accs)
        else:
            # Final accuracy on all seen classes
            y_values = [task_accs[t][-1] if task_accs[t] else 0 for t in sorted(task_accs.keys())]
        
        x_values = list(range(len(y_values)))
        y_values = [v * 100 for v in y_values]  # Convert to percentage
        
        ax.plot(x_values, y_values, 
                color=colors[idx], 
                marker=markers[idx],
                linestyle=linestyles[idx % len(linestyles)],
                label=strategy,
                markersize=6,
                linewidth=1.5)
        
        # Add std bands if available
        std_key = f"{strategy}_std"
        if show_std and std_key in results:
            std_accs = results[std_key]
            if metric == 'aia':
                std_values = compute_average_incremental_accuracy(std_accs)
            else:
                std_values = [std_accs[t][-1] if std_accs[t] else 0 for t in sorted(std_accs.keys())]
            
            std_values = [v * 100 for v in std_values]
            y_lower = [y - s for y, s in zip(y_values, std_values)]
            y_upper = [y + s for y, s in zip(y_values, std_values)]
            ax.fill_between(x_values, y_lower, y_upper, 
                           color=colors[idx], alpha=0.15)
    
    ax.set_xlabel('Task $t$')
    ylabel = 'Average Incremental Accuracy (%)' if metric == 'aia' else 'Final Accuracy (%)'
    ax.set_ylabel(ylabel)
    
    if title is None:
        title = 'Incremental Learning Performance'
    ax.set_title(title)
    
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'$T_{{{i}}}$' for i in x_values])
    ax.set_ylim([0, 100])
    ax.legend(loc='lower left', fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_forgetting_comparison(
    results: Dict[str, Dict[int, List[float]]],
    figsize: tuple = None,
    save_path: Optional[str] = None,
    title: str = None
) -> plt.Figure:
    """
    Plot Forgetting Measure comparison across strategies.
    
    Shows bar chart of average forgetting and per-task forgetting.
    
    Args:
        results: {strategy_name: {task_id: [acc_task_0, ..., acc_task_t]}}
        figsize: Figure size
        save_path: Optional path to save
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('single', aspect=0.75)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = list(results.keys())
    colors = get_ieee_colors(len(strategies))
    
    forgetting_data = {}
    for strategy in strategies:
        task_accs = results[strategy]
        forgetting = compute_forgetting_measure(task_accs)
        forgetting_data[strategy] = {
            'values': forgetting,
            'avg': np.mean(forgetting) * 100,
            'std': np.std(forgetting) * 100
        }
    
    # Bar chart of average forgetting
    x = np.arange(len(strategies))
    avgs = [forgetting_data[s]['avg'] for s in strategies]
    stds = [forgetting_data[s]['std'] for s in strategies]
    
    bars = ax.bar(x, avgs, yerr=stds, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, avgs):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Average Forgetting (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha='right')
    
    if title is None:
        title = 'Forgetting Measure Comparison'
    ax.set_title(title)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylim(bottom=min(0, min(avgs) - 5))
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_task_accuracy_heatmap(
    task_accuracies: Dict[int, List[float]],
    strategy_name: str = None,
    figsize: tuple = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Task-wise Accuracy Heatmap.
    
    Rows: Evaluation after task t
    Cols: Accuracy on task i
    
    Standard visualization showing catastrophic forgetting pattern.
    
    Args:
        task_accuracies: {task_id: [acc_task_0, ..., acc_task_t]}
        strategy_name: Name of strategy for title
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    n_tasks = max(task_accuracies.keys()) + 1
    
    # Build accuracy matrix
    acc_matrix = np.zeros((n_tasks, n_tasks))
    acc_matrix[:] = np.nan  # Fill with NaN for upper triangle
    
    for t in range(n_tasks):
        if t in task_accuracies:
            for i, acc in enumerate(task_accuracies[t]):
                if i <= t:
                    acc_matrix[t, i] = acc * 100
    
    if figsize is None:
        size = max(3.5, n_tasks * 0.6)
        figsize = (size, size * 0.9)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap: white for NaN, blue-red for values
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='white')
    
    im = ax.imshow(acc_matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Add text annotations
    for t in range(n_tasks):
        for i in range(n_tasks):
            if not np.isnan(acc_matrix[t, i]):
                text_color = 'white' if acc_matrix[t, i] < 50 else 'black'
                ax.text(i, t, f'{acc_matrix[t, i]:.1f}',
                       ha='center', va='center', fontsize=7, color=text_color)
    
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f'$T_{{{i}}}$' for i in range(n_tasks)])
    ax.set_yticklabels([f'After $T_{{{i}}}$' for i in range(n_tasks)])
    
    ax.set_xlabel('Task (Evaluated)')
    ax.set_ylabel('Task (Trained)')
    
    title = f'Task-wise Accuracy Matrix'
    if strategy_name:
        title += f' ({strategy_name})'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_old_new_accuracy(
    results: Dict[str, Dict[int, Dict[str, float]]],
    figsize: tuple = None,
    save_path: Optional[str] = None,
    title: str = None
) -> plt.Figure:
    """
    Plot Old Classes vs New Classes Accuracy.
    
    Standard FCIL plot showing the stability-plasticity tradeoff.
    
    Args:
        results: {strategy: {task_id: {'old': acc, 'new': acc, 'all': acc}}}
        figsize: Figure size
        save_path: Optional path to save
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('double', aspect=0.45)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    strategies = list(results.keys())
    colors = get_ieee_colors(len(strategies))
    markers = get_ieee_markers(len(strategies))
    
    # Plot 1: Old class accuracy
    ax1 = axes[0]
    for idx, strategy in enumerate(strategies):
        task_data = results[strategy]
        tasks = sorted(task_data.keys())
        old_accs = [task_data[t].get('old', 0) * 100 for t in tasks]
        
        # Old accuracy only from task 1 onwards
        if len(tasks) > 1:
            ax1.plot(tasks[1:], old_accs[1:],
                    color=colors[idx], marker=markers[idx],
                    label=strategy, markersize=5, linewidth=1.5)
    
    ax1.set_xlabel('Task $t$')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Old Classes')
    ax1.legend(fontsize=6, loc='lower left')
    ax1.set_ylim([0, 100])
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: New class accuracy
    ax2 = axes[1]
    for idx, strategy in enumerate(strategies):
        task_data = results[strategy]
        tasks = sorted(task_data.keys())
        new_accs = [task_data[t].get('new', 0) * 100 for t in tasks]
        
        ax2.plot(tasks, new_accs,
                color=colors[idx], marker=markers[idx],
                label=strategy, markersize=5, linewidth=1.5)
    
    ax2.set_xlabel('Task $t$')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) New Classes')
    ax2.legend(fontsize=6, loc='lower left')
    ax2.set_ylim([0, 100])
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 3: Overall accuracy
    ax3 = axes[2]
    for idx, strategy in enumerate(strategies):
        task_data = results[strategy]
        tasks = sorted(task_data.keys())
        all_accs = [task_data[t].get('all', 0) * 100 for t in tasks]
        
        ax3.plot(tasks, all_accs,
                color=colors[idx], marker=markers[idx],
                label=strategy, markersize=5, linewidth=1.5)
    
    ax3.set_xlabel('Task $t$')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('(c) All Classes')
    ax3.legend(fontsize=6, loc='lower left')
    ax3.set_ylim([0, 100])
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    if title:
        fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_bwt_fwt_comparison(
    results: Dict[str, Dict[int, List[float]]],
    figsize: tuple = None,
    save_path: Optional[str] = None,
    title: str = None
) -> plt.Figure:
    """
    Plot Backward Transfer (BWT) and Forward Transfer (FWT) comparison.
    
    Args:
        results: {strategy_name: {task_id: [acc_task_0, ..., acc_task_t]}}
        figsize: Figure size
        save_path: Optional path to save
        title: Optional custom title
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('single', aspect=0.8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = list(results.keys())
    colors = get_ieee_colors(len(strategies))
    
    bwt_values = []
    fwt_values = []
    
    for strategy in strategies:
        task_accs = results[strategy]
        bwt = compute_backward_transfer(task_accs) * 100
        fwt = compute_forward_transfer(task_accs) * 100
        bwt_values.append(bwt)
        fwt_values.append(fwt)
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bwt_values, width, label='BWT', 
                   color=colors[0], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, fwt_values, width, label='FWT',
                   color=colors[1] if len(colors) > 1 else '#808080', 
                   edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bars, values in [(bars1, bwt_values), (bars2, fwt_values)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            y_pos = height + 0.5 if height >= 0 else height - 2
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2 if height >= 0 else -8),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Transfer (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha='right')
    
    if title is None:
        title = 'Backward and Forward Transfer'
    ax.set_title(title)
    ax.legend(loc='best', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


# ==============================================================================
# ADVANCED FCIL PLOTS
# ==============================================================================

def plot_class_accuracy_evolution(
    class_accuracies: Dict[int, Dict[int, float]],
    n_tasks: int,
    classes_per_task: List[int],
    figsize: tuple = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-class accuracy evolution across tasks.
    
    Shows how accuracy of each class changes as new tasks are learned.
    
    Args:
        class_accuracies: {task_id: {class_id: accuracy}}
        n_tasks: Total number of tasks
        classes_per_task: List of number of classes in each task
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('double', aspect=0.5)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Assign colors by task
    task_colors = get_ieee_colors(n_tasks)
    
    # Track which classes belong to which task
    class_to_task = {}
    class_idx = 0
    for t, n_cls in enumerate(classes_per_task):
        for _ in range(n_cls):
            class_to_task[class_idx] = t
            class_idx += 1
    
    total_classes = sum(classes_per_task)
    
    for cls in range(total_classes):
        task = class_to_task[cls]
        
        # Get accuracy evolution for this class
        x_vals = []
        y_vals = []
        
        for t in range(task, n_tasks):
            if t in class_accuracies and cls in class_accuracies[t]:
                x_vals.append(t)
                y_vals.append(class_accuracies[t][cls] * 100)
        
        if x_vals:
            ax.plot(x_vals, y_vals, color=task_colors[task], 
                   alpha=0.6, linewidth=1, marker='.')
    
    # Add task legend
    patches = [mpatches.Patch(color=task_colors[t], label=f'Task {t} classes') 
               for t in range(n_tasks)]
    ax.legend(handles=patches, loc='lower left', fontsize=7)
    
    ax.set_xlabel('Task $t$')
    ax.set_ylabel('Class Accuracy (%)')
    ax.set_title('Per-Class Accuracy Evolution')
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f'$T_{{{i}}}$' for i in range(n_tasks)])
    ax.set_ylim([0, 100])
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_convergence_per_task(
    training_history: Dict[int, Dict[str, List[float]]],
    figsize: tuple = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot convergence curves for each task.
    
    Shows training dynamics across tasks - useful for analyzing 
    learning rate adaptation and task difficulty.
    
    Args:
        training_history: {task_id: {'train_loss': [...], 'test_acc': [...]}}
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    n_tasks = len(training_history)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (3.5 * n_cols, 2.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_tasks == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = get_ieee_colors(2)
    
    for task_id in range(n_tasks):
        row = task_id // n_cols
        col = task_id % n_cols
        ax = axes[row, col]
        
        if task_id in training_history:
            history = training_history[task_id]
            
            # Plot loss on left y-axis
            rounds = range(1, len(history.get('train_loss', [])) + 1)
            if 'train_loss' in history:
                ax.plot(rounds, history['train_loss'], 
                       color=colors[0], label='Loss', linewidth=1)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Loss', color=colors[0])
            ax.tick_params(axis='y', labelcolor=colors[0])
            
            # Plot accuracy on right y-axis
            if 'test_acc' in history:
                ax2 = ax.twinx()
                acc_rounds = range(1, len(history['test_acc']) + 1)
                acc_vals = [a * 100 for a in history['test_acc']]
                ax2.plot(acc_rounds, acc_vals,
                        color=colors[1], label='Acc', linewidth=1, linestyle='--')
                ax2.set_ylabel('Accuracy (%)', color=colors[1])
                ax2.tick_params(axis='y', labelcolor=colors[1])
                ax2.set_ylim([0, 100])
        
        ax.set_title(f'Task {task_id}', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_tasks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle('Convergence per Task', fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_multi_strategy_radar(
    results: Dict[str, Dict[str, float]],
    figsize: tuple = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Radar/Spider chart comparing multiple strategies across metrics.
    
    Standard visualization for comprehensive strategy comparison.
    
    Args:
        results: {strategy: {'AIA': val, 'Forgetting': val, 'BWT': val, ...}}
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = (5, 5)
    
    strategies = list(results.keys())
    metrics = list(results[strategies[0]].keys())
    n_metrics = len(metrics)
    
    # Normalize metrics to [0, 1] for radar chart
    all_values = {m: [] for m in metrics}
    for strategy in strategies:
        for m in metrics:
            all_values[m].append(results[strategy][m])
    
    # Normalize (higher is better, except for Forgetting)
    normalized = {s: {} for s in strategies}
    for m in metrics:
        vals = all_values[m]
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v if max_v != min_v else 1
        
        for s in strategies:
            norm_val = (results[s][m] - min_v) / range_v
            # Invert forgetting (lower is better)
            if 'Forgetting' in m or 'forgetting' in m:
                norm_val = 1 - norm_val
            normalized[s][m] = norm_val
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = get_ieee_colors(len(strategies))
    
    for idx, strategy in enumerate(strategies):
        values = [normalized[strategy][m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, color=colors[idx], linewidth=1.5, label=strategy)
        ax.fill(angles, values, color=colors[idx], alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim([0, 1])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)
    ax.set_title('Multi-Metric Comparison', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


def plot_communication_efficiency(
    comm_data: Dict[str, Dict[str, Union[List[float], float]]],
    figsize: tuple = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot communication efficiency comparison.
    
    Useful for FCIL papers focusing on communication overhead.
    
    Args:
        comm_data: {strategy: {'bytes_per_round': [...], 'total_bytes': float, 
                              'compression_ratio': float}}
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('double', aspect=0.45)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    strategies = list(comm_data.keys())
    colors = get_ieee_colors(len(strategies))
    
    # Plot 1: Bytes per round
    ax1 = axes[0]
    for idx, strategy in enumerate(strategies):
        data = comm_data[strategy]
        if 'bytes_per_round' in data:
            rounds = range(1, len(data['bytes_per_round']) + 1)
            # Convert to MB
            bytes_mb = [b / (1024 * 1024) for b in data['bytes_per_round']]
            ax1.plot(rounds, bytes_mb, color=colors[idx], 
                    label=strategy, linewidth=1.5)
    
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Data Transmitted (MB)')
    ax1.set_title('(a) Communication per Round')
    ax1.legend(fontsize=7)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Total communication and compression
    ax2 = axes[1]
    x = np.arange(len(strategies))
    
    total_bytes = [comm_data[s].get('total_bytes', 0) / (1024 * 1024) 
                   for s in strategies]  # Convert to MB
    
    bars = ax2.bar(x, total_bytes, color=colors, edgecolor='black', linewidth=0.8)
    
    # Add compression ratio labels
    for idx, (bar, strategy) in enumerate(zip(bars, strategies)):
        height = bar.get_height()
        ratio = comm_data[strategy].get('compression_ratio', 1.0)
        ax2.annotate(f'{ratio:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)
    
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Total Communication (MB)')
    ax2.set_title('(b) Total Communication')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=30, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig


# ==============================================================================
# COMPREHENSIVE FCIL REPORT
# ==============================================================================

def generate_fcil_report(
    results: Dict[str, Dict[int, List[float]]],
    old_new_accs: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None,
    training_history: Optional[Dict[int, Dict[str, List[float]]]] = None,
    output_dir: str = 'fcil_results',
    prefix: str = ''
) -> Dict[str, float]:
    """
    Generate comprehensive FCIL visualization report.
    
    Creates all standard plots for FCIL paper:
    1. Incremental accuracy curve
    2. Forgetting comparison
    3. BWT/FWT comparison  
    4. Task accuracy heatmaps
    5. Old vs New accuracy (if provided)
    6. Convergence per task (if provided)
    
    Args:
        results: {strategy: {task_id: [acc_task_0, ..., acc_task_t]}}
        old_new_accs: Optional old/new accuracy data
        training_history: Optional training history per task
        output_dir: Directory to save plots
        prefix: Filename prefix
    
    Returns:
        Dict of computed metrics for each strategy
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_summary = {}
    
    # Compute metrics for each strategy
    for strategy, task_accs in results.items():
        aia = compute_average_incremental_accuracy(task_accs)
        forgetting = compute_forgetting_measure(task_accs)
        bwt = compute_backward_transfer(task_accs)
        fwt = compute_forward_transfer(task_accs)
        
        metrics_summary[strategy] = {
            'Final_AIA': aia[-1] * 100 if aia else 0,
            'Avg_Forgetting': np.mean(forgetting) * 100 if forgetting else 0,
            'BWT': bwt * 100,
            'FWT': fwt * 100,
            'Last_Task_Acc': task_accs[max(task_accs.keys())][-1] * 100 
                            if task_accs else 0
        }
    
    # Generate plots
    print("📊 Generating FCIL report...")
    
    # 1. Incremental accuracy curve
    fig1 = plot_incremental_accuracy_curve(
        results, 
        save_path=os.path.join(output_dir, f'{prefix}incremental_accuracy.png')
    )
    plt.close(fig1)
    
    # 2. Forgetting comparison
    fig2 = plot_forgetting_comparison(
        results,
        save_path=os.path.join(output_dir, f'{prefix}forgetting_comparison.png')
    )
    plt.close(fig2)
    
    # 3. BWT/FWT comparison
    fig3 = plot_bwt_fwt_comparison(
        results,
        save_path=os.path.join(output_dir, f'{prefix}bwt_fwt.png')
    )
    plt.close(fig3)
    
    # 4. Task accuracy heatmaps for each strategy
    for strategy, task_accs in results.items():
        fig4 = plot_task_accuracy_heatmap(
            task_accs,
            strategy_name=strategy,
            save_path=os.path.join(output_dir, f'{prefix}heatmap_{strategy}.png')
        )
        plt.close(fig4)
    
    # 5. Old vs New accuracy (if provided)
    if old_new_accs:
        fig5 = plot_old_new_accuracy(
            old_new_accs,
            save_path=os.path.join(output_dir, f'{prefix}old_new_accuracy.png')
        )
        plt.close(fig5)
    
    # 6. Convergence per task (if provided)
    if training_history:
        fig6 = plot_convergence_per_task(
            training_history,
            save_path=os.path.join(output_dir, f'{prefix}convergence.png')
        )
        plt.close(fig6)
    
    # 7. Radar chart
    radar_data = {s: m for s, m in metrics_summary.items()}
    fig7 = plot_multi_strategy_radar(
        radar_data,
        save_path=os.path.join(output_dir, f'{prefix}radar_comparison.png')
    )
    plt.close(fig7)
    
    # Export metrics as LaTeX table
    _export_fcil_latex_table(metrics_summary, 
                             os.path.join(output_dir, f'{prefix}metrics_table.tex'))
    
    print(f"✅ FCIL report saved to: {output_dir}")
    
    return metrics_summary


def _export_fcil_latex_table(
    metrics: Dict[str, Dict[str, float]],
    output_path: str
) -> str:
    """
    Export FCIL metrics as LaTeX table.
    
    Args:
        metrics: {strategy: {metric_name: value}}
        output_path: Path to save LaTeX file
    
    Returns:
        LaTeX table string
    """
    strategies = list(metrics.keys())
    metric_names = list(metrics[strategies[0]].keys())
    
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Federated Class-Incremental Learning Results}")
    latex.append(r"\label{tab:fcil_results}")
    
    # Column format
    col_fmt = 'l' + 'c' * len(metric_names)
    latex.append(r"\begin{tabular}{" + col_fmt + "}")
    latex.append(r"\hline")
    
    # Header
    header = r"\textbf{Method}"
    for m in metric_names:
        # Format metric name for LaTeX
        m_fmt = m.replace('_', ' ').replace('%', r'\%')
        header += f" & \\textbf{{{m_fmt}}}"
    header += r" \\"
    latex.append(header)
    latex.append(r"\hline")
    
    # Data rows
    for strategy in strategies:
        row = strategy
        for m in metric_names:
            val = metrics[strategy][m]
            # Format based on metric type
            if 'Forgetting' in m:
                row += f" & {val:.2f}$\\downarrow$"
            elif 'BWT' in m:
                row += f" & {val:+.2f}"
            else:
                row += f" & {val:.2f}"
        row += r" \\"
        latex.append(row)
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    latex_str = "\n".join(latex)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"📄 Saved LaTeX table: {output_path}")
    
    return latex_str


def _save_figure(fig: plt.Figure, save_path: str) -> None:
    """Save figure as PNG and PDF."""
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"📊 Saved: {save_path} & {pdf_path}")


# ==============================================================================
# STATISTICAL ANALYSIS PLOTS
# ==============================================================================

def plot_statistical_comparison(
    results: Dict[str, List[float]],
    metric_name: str = 'Accuracy',
    figsize: tuple = None,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """
    Box plot with statistical significance tests.
    
    Standard for comparing methods across multiple runs.
    
    Args:
        results: {strategy: [run1_val, run2_val, ...]}
        metric_name: Name of metric being compared
        figsize: Figure size
        save_path: Optional path to save
    
    Returns:
        Tuple of (Figure, p-value matrix)
    """
    set_ieee_style()
    
    if figsize is None:
        figsize = get_ieee_figsize('single', aspect=0.9)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = list(results.keys())
    data = [results[s] for s in strategies]
    colors = get_ieee_colors(len(strategies))
    
    # Box plot
    bp = ax.boxplot(data, patch_artist=True, labels=strategies)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for idx, (strategy, values) in enumerate(results.items()):
        x = np.random.normal(idx + 1, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.5, color='black', s=15, zorder=3)
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Distribution Across Runs')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Compute pairwise t-tests
    p_values = {}
    for i, s1 in enumerate(strategies):
        p_values[s1] = {}
        for j, s2 in enumerate(strategies):
            if i < j:
                _, p = stats.ttest_ind(results[s1], results[s2])
                p_values[s1][s2] = p
    
    plt.tight_layout()
    
    if save_path:
        _save_figure(fig, save_path)
    
    return fig, p_values
