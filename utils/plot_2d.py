import numpy as np
import matplotlib.pyplot as plt

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