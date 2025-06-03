import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(vector: np.ndarray, bins: int = None, title: str = "vector", xlabel: str = "vector"):
    """
    Plots a histogram for a vector.
    
    Args:
        vector (np.ndarray): A (N,) array of numbers.
        bins (int): Number of histogram bins.
        title (str): Title of the plot.
        xlabel (str): Label of the x-axis
    """
    
    if bins is None:
        # Automatically choose number of bins based on data spread
        data_range = vector.max() - vector.min()
        if data_range == 0:
            bins = 1  # all vectors are identical in length
        else:
            bins = min(50, max(1, int(len(vector) / 20)))  # heuristic fallback

    
    plt.figure(figsize=(6, 4))
    plt.hist(vector, bins=bins, color='skyblue', edgecolor='black', alpha=0.75)
    plt.axvline(1.0, color='black', linestyle='--', label='Expected length = 1')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()