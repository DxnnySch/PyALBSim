import matplotlib.pyplot as plt

def plot_scatter_2d(x, y, xlabel = "X-Axis", ylabel = "Y-Axis"):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=2, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.grid(True)
    plt.show()