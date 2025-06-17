import matplotlib.pyplot as plt

def plot_scatter_2d(x, y, c=None, xlabel="X-Axis", ylabel="Y-Axis", colorbar_label="Color Axis"):
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(x, y, c=c, s=2, alpha=0.6, cmap='viridis')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.grid(True)
    
    if c is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label(colorbar_label)

    plt.show()
