import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


def vis_data(X,y):
    '''X must be 2D and y must be 1D'''
    y_numeric = LabelEncoder().fit_transform(y)

    # Plotting settings
    fig, ax = plt.subplots(figsize=(8, 8))
    #x_min, x_max, y_min, y_max = -3, 3, -3, 3
    #ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_numeric,  cmap="viridis", s=150, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title("Samples in two-dimensional feature space")
    _ = plt.show()


