from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from alb_sim.photon_mapping.photon_type import PhotonType
from alb_sim.plotting.format_value_for_plot import format_value
from alb_sim.utils.types import Array


def plot_stacked_waveform(
    data: dict[PhotonType, Array],
    title: str = "Full Waveform (Stacked)",
    xlabel: str = "Step / Distance",
    ylabel: str = "Photon contribution",
    xlim: Optional[tuple[int, int]] = None,
    layer_lines: Optional[list[float]] = None,
    params: Optional[dict] = None,
    save_path: Optional[str] = None,
    padding: int = 50,
    show: bool = True,
):
    """
    Plots a line diagram, shows it full-screen, and optionally saves it.
    Optionally adds parameter info from a dictionary into the image.
    Automatically calculates x-limits based on non-zero y-values if xlim is not provided.

    Args:
        data (np.ndarray): A (N,) array of numbers.
        title (str): Title of the plot.
        xlabel (str): Label of the x-axis.
        ylabel (str): Label of the y-axis.
        xlim (tuple[int, int], optional): Range of x-values to display, e.g. (100, 200).
        layer_lines (list[float], optional): x-values where the layers change.
        params (dict, optional): Parameters to annotate in the plot.
        save_path (str, optional): If given, the plot is saved to this path.
        padding (int): Number of samples to add before and after non-zero region (default: 50).
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(next(iter(data.values()))))
    labels = [pt.name for pt in data]
    values = np.vstack(list(data.values()))
    plt.stackplot(x, values, labels=labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.legend()
    ax.grid(True)

    if layer_lines is not None:
        x_lines = np.array(layer_lines) * 2
        ax.vlines(x_lines, 0, np.max(np.sum(list(data.values()), axis=0)))

    # Calculate x-limits based on non-zero y-values if not provided
    if xlim is None:
        # Find indices where data is non-zero
        non_zero_indices = np.where(np.sum(list(data.values()), axis=0) != 0)[0]

        if len(non_zero_indices) > 0:
            # Get the first and last non-zero indices
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]

            # Add padding before and after
            start_idx = max(0, first_non_zero - padding)
            end_idx = min(
                len(np.sum(list(data.values()), axis=0)) - 1, last_non_zero + padding
            )

            ax.set_xlim(start_idx, end_idx)
        # If all values are zero, keep default x-limits

    if xlim is not None:
        ax.set_xlim(*xlim)

    # --- Add parameter info as text ---
    if params is not None:
        # Format keys and values
        formatted_items = [(str(k), format_value(v)) for k, v in params.items()]
        max_key_len = max(len(k) for k, _ in formatted_items)
        max_val_len = max(len(v) for _, v in formatted_items)
        # Pad keys (right) and values (left)
        param_text = "\n".join(
            [f"{k:<{max_key_len}} : {v:>{max_val_len}}" for k, v in formatted_items]
        )
        # Place it in the top-right corner inside the axes
        ax.text(
            0.99,
            0.99,
            param_text,
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            color="dimgray",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
        )

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.075, right=0.975, top=0.945)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved to {save_path}")

    if show:
        plt.show()
