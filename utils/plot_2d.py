import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def plot_2d(data: np.ndarray, title: str = "vector", xlabel: str = "X-Axis", ylabel: str = "Y-Axis"):
    """
    Plots a line diagram.
    
    Args:
        data (np.ndarray): A (N,) array of numbers.
        title (str): Title of the plot.
        xlabel (str): Label of the x-axis
        ylabel (str): Label of the y-axis
    """

    
    plt.figure(figsize=(6, 4))
    plt.plot(data, color='skyblue', alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.075, right=0.975, top=0.945)
    plt.show()


def format_value(v):
    """Nicely formats numeric values for display."""
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"  # add commas for large integers
    elif isinstance(v, (float, np.floating)):
        # Format floats depending on magnitude
        if abs(v) >= 1e3 or abs(v) < 1e-3:
            return f"{v:.3e}"  # scientific notation
        else:
            return f"{v:.3f}".rstrip('0').rstrip('.')  # up to 3 decimals, remove trailing zeros
    else:
        return str(v)

def plot_2d_better(
    data: np.ndarray,
    title: str = "vector",
    xlabel: str = "X-Axis",
    ylabel: str = "Y-Axis",
    xlim: Tuple[int, int] | None = None,
    params: dict | None = None,
    save_path: str | None = None,
    padding: int = 50,
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
        params (dict, optional): Parameters to annotate in the plot.
        save_path (str, optional): If given, the plot is saved to this path.
        padding (int): Number of samples to add before and after non-zero region (default: 50).
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Try fullscreen
    mng = plt.get_current_fig_manager()
    try:
        mng.full_screen_toggle()
    except Exception:
        fig.set_size_inches(19.2, 10.8)  # fallback fullscreen

    ax.plot(data, color='skyblue', alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.legend()
    ax.grid(True)

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
        param_text = "\n".join([
            f"{k:<{max_key_len}} : {v:>{max_val_len}}"
            for k, v in formatted_items
        ])
        # Place it in the top-right corner inside the axes
        ax.text(
            0.99, 0.99, param_text,
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            color="dimgray",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
        )

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.075, right=0.975, top=0.945)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")

    # plt.show()