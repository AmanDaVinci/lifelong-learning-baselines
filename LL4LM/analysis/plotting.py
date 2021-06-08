import numpy as np
import matplotlib.pyplot as plt
from LL4LM.analysis.processing import (
    get_train_accuracies, 
    get_test_accuracies,
    get_gradient_measurements
)


plt.style.use("seaborn-talk")

def draw_canvas(stream, ymin=0, ymax=1):
    fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True, dpi=300)
    boundaries = np.cumsum([dataset_examples for _, dataset_examples in stream])
    for boundary in boundaries:
        ax.vlines(x=boundary, ymin=ymin, ymax=ymax, linestyle="dashed", color="gray")
    ax.set_xticks(boundaries)
    ax.set_xticklabels(boundaries, rotation="vertical")
    top_xaxis = ax.secondary_xaxis("top")
    top_xaxis.set_xticks(boundaries)
    top_xaxis.set_xticklabels([name for name, _ in stream], rotation="vertical")
    return fig, ax, boundaries

def plot_lifelong_curve(name, stream, logs, multitask_logs, unitask_logs, 
                        training=True, testing=False, testing_detailed=False):
    fig, ax, boundaries = draw_canvas(stream)
    if training:
        exp_accuracies = get_train_accuracies(logs, stream, rolling_window=20)
        mtl_accuracies = get_train_accuracies(multitask_logs, stream, rolling_window=20)
        utl_accuracies = get_train_accuracies(unitask_logs, stream, rolling_window=20)
        ax.plot(exp_accuracies.index, exp_accuracies.values, 
                label=f"{name} Training", color="tab:orange")
        ax.plot(mtl_accuracies.index, mtl_accuracies.values, 
                 label="Multi-task Training", color="tab:pink", alpha=0.5)
        ax.plot(utl_accuracies.index, utl_accuracies.values, 
                 label="Uni-task Training", color="blue", alpha=0.3)
    elif testing:
        exp_accuracies = get_test_accuracies(logs, stream)
        mtl_accuracies = get_test_accuracies(multitask_logs, stream)
        utl_accuracies = get_test_accuracies(unitask_logs, stream, unitask=True)
        if testing_detailed:
            dataset_colors = plt.cm.Set1(range(len(stream)))
            for (dataset_name, _), boundary, color in zip(stream, boundaries, dataset_colors):
                ax.plot(exp_accuracies.index, exp_accuracies[dataset_name], color=color, alpha=0.5)
                ax.vlines(x=boundary, ymin=0, ymax=1, linestyle="dashed", color=color)
                ax.scatter(x=boundary, y=utl_accuracies[dataset_name], color=color, marker="o")
        else:
            ax.plot(exp_accuracies.index, exp_accuracies.mean(axis=1), 
                    label=f"{name} Testing", color="tab:orange")
            ax.plot(mtl_accuracies.index, mtl_accuracies.mean(axis=1), 
                    label="Multi-task Testing", color="tab:pink", alpha=0.5)
            for (dataset_name, _), boundary in zip(stream, boundaries):
                ax.scatter(x=boundary, y=utl_accuracies[dataset_name], 
                           label="Uni-task Testing", color="blue", alpha=0.5, marker="o")
    ax.set_xlabel("Streaming Examples")
    ax.set_ylabel("Accuracy")
    ax.grid(False, axis='x')
    ax.grid(True, axis='y')
    if not testing_detailed:
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], loc="lower left")
    return fig

def plot_gradient_interference(name, stream, logs, multitask_logs):
    fig, ax, _ = draw_canvas(stream)
    exp_intf = get_gradient_measurements(logs, stream, rolling_window=10)
    mtl_intf = get_gradient_measurements(multitask_logs, stream, rolling_window=10)
    ax.plot(exp_intf.index, exp_intf.values, label=name, color="tab:orange")
    ax.plot(mtl_intf.index, mtl_intf.values, label="Multi-task", color="tab:pink", alpha=0.5)
    ax.set_xlabel("Streaming Examples")
    ax.set_ylabel("Gradient Interference")
    ax.grid(False, axis='x')
    ax.grid(True, axis='y')
    ax.legend(loc="upper right")
    return fig

def plot_gradient_overlap(name, stream, logs, multitask_logs):
    fig, ax, _ = draw_canvas(stream, ymin=0, ymax=0)
    exp_overlap = get_gradient_measurements(logs, stream, measurement="overlap", rolling_window=10)
    mtl_overlap = get_gradient_measurements(multitask_logs, stream, measurement="overlap", rolling_window=10)
    ax.plot(exp_overlap.index, exp_overlap.values, label=name, color="tab:orange")
    ax.plot(mtl_overlap.index, mtl_overlap.values, label="Multi-task", color="tab:pink", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Streaming Examples")
    ax.set_ylabel("Gradient Overlap (Log Scale)")
    ax.grid(False, axis='x')
    ax.grid(True, axis='y')
    ax.legend(loc="upper right")
    return fig
