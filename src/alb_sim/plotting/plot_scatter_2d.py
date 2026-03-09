import matplotlib.pyplot as plt


def plot_scatter_2d(
    x, y, c=None, xlabel="X-Axis", ylabel="Y-Axis", colorbar_label="Color Axis"
):
    """
    Plot a simple 2D scatter with optional colour coding.

    Parameters
    ----------
    x, y : array_like
        Coordinates of points to scatter.
    c : array_like, optional
        Optional colour values for each point.
    xlabel, ylabel : str, optional
        Axis labels.
    colorbar_label : str, optional
        Label for the colourbar when ``c`` is provided.
    """
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(x, y, c=c, s=2, alpha=0.6, cmap="viridis")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.grid(True)

    if c is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label(colorbar_label)

    plt.show()
