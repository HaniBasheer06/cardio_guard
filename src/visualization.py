"""
Visualization Module - Bridges interactive Plotly functions and publication Matplotlib plots.
"""

from utils.visualization import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix_heatmap(cm: list) -> plt.Figure:
    """Plots a 2x2 confusion matrix heatmap with counts and percentages."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm_array = np.array(cm)
    total = np.sum(cm_array)

    cax = ax.matshow(cm_array, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, shrink=0.8)

    classes = ["No Disease (0)", "Disease (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Class", labelpad=10, fontweight="bold")
    ax.set_ylabel("Actual Class", labelpad=10, fontweight="bold")
    ax.set_title("Test Confusion Matrix (Held-out 20%)", pad=15)

    cell_labels = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"],
    ]

    for i in range(2):
        for j in range(2):
            val = cm_array[i, j]
            pct = (val / total) * 100
            label = cell_labels[i][j]
            text_color = "white" if val > (total / 3) else "#1e293b"
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)\n{label}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    return fig
