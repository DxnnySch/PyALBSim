from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from alb_sim.plotting.format_value_for_plot import format_value
from alb_sim.utils.types import Array


def plot_waveform(
    data: Array,
    title: str = "Full Waveform",
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
    Plot a 1D waveform with optional auto-zoom and parameter annotations.

    Parameters
    ----------
    data : Array
        Waveform samples to plot.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    xlim : tuple[int, int], optional
        Explicit x-axis limits; if None, limits are inferred from non-zero samples.
    layer_lines : list[float], optional
        Optional vertical lines (e.g. water-layer boundaries) to overlay.
    params : dict, optional
        Dictionary of parameters to render as a text box inside the figure.
    save_path : str, optional
        If provided, file path to save the figure.
    padding : int, optional
        Number of samples for padding around the non-zero region when auto-zooming.
    show : bool, optional
        If True, display the figure interactively.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data)
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
        non_zero_indices = np.where(data != 0)[0]

        if len(non_zero_indices) > 0:
            # Get the first and last non-zero indices
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]

            # Add padding before and after
            start_idx = max(0, first_non_zero - padding)
            end_idx = min(len(data) - 1, last_non_zero + padding)

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
