import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay



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


def conf_matrix(y_train, y_pred, vis=True):
    ''' Plot confusion matrix '''
    cm = confusion_matrix(y_train, y_pred)
    # print("Confusion Matrix:\n", cm)
    labels = np.unique(y_train)

    if vis:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Greens")
        ax.set_title("Confusion Matrix")
        plt.show()

    return cm

def recall(cm, y_train, vis=True):
    # Compute recall (sensitivity) per class [TP/[TP+FN]]
    labels = np.unique(y_train)
    per_class_accuracy = np.divide(
        cm.diagonal(), cm.sum(axis=1), out=np.zeros_like(cm.diagonal(), dtype=float), where=cm.sum(axis=1) != 0
    )

    print("\nRecall (Per-Class Accuracy):")
    for label, acc in zip(labels, per_class_accuracy):
        print(f"Class {label}: {acc:.3f}")

    if vis:
        # Determine the best class
        best_class_index = np.argmax(per_class_accuracy)

        # Assign colors: Green for best, Blue for others
        colors = ['blue'] * len(labels)
        colors[best_class_index] = 'green'  # Highlight best class

        # Plot bar chart
        plt.figure(figsize=(6, 4))
        plt.bar(labels.astype(str), per_class_accuracy, color=colors)

        # Formatting
        plt.xlabel("Class")
        plt.ylabel("Recall (Per-Class Accuracy)")
        plt.title("Recall Per Class")
        plt.ylim(0, 1)  # Recall is between 0 and 1
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Annotate best class
        plt.text(best_class_index, per_class_accuracy[best_class_index] + 0.02, "Best",
                 ha='center', fontsize=10, fontweight='bold', color='green')

        plt.show()

def classification_accuracy(cm, y_train):
    # Compute Classification accuracy (CA) ORANGE feature
    # using (TP + TN) / Total Samples
    total_samples = np.sum(cm)  # Total number of samples
    per_class_CA = []
    for i in range(len(cm)):
        TP = cm[i, i]  # True Positives
        FN = np.sum(cm[i, :]) - TP  # False Negatives
        FP = np.sum(cm[:, i]) - TP  # False Positives
        TN = total_samples - (TP + FN + FP)  # True Negatives

        per_class_CA.append((TP + TN) / total_samples)

    # Print results
    labels = np.unique(y_train)
    print("\nClassification accuracy (CA):")
    for label, ca in zip(labels, per_class_CA):
        print(f"Class {label}: {ca:.3f}")

