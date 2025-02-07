import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score




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

def ROC_analysis_target(y_score, y_train, target):
    lb = LabelBinarizer()
    y_onehot_true = lb.fit_transform(y_train)

    # Select class of interest for one-vs-rest ROC
    class_of_interest = target
    class_id = np.flatnonzero(lb.classes_ == class_of_interest)[0]
    print(f"Class ID for {class_of_interest}: {class_id}")

    # Plot ROC Curve
    display = RocCurveDisplay.from_predictions(
        y_onehot_true[:, class_id],  # True labels (one-hot)
        y_score[:, class_id],  # Model scores (probabilities or decision function)
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
        plot_chance_level=True,
        despine=True,
    )

    # Improve Plot Appearance
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"One-vs-Rest ROC Curve:\n{class_of_interest} vs All",
    )
    plt.show()

def ROC_analysis_all_targets(y_score, y_train):
    # Convert y_train to one-hot encoding
    lb = LabelBinarizer()
    y_onehot_true = lb.fit_transform(y_train)
    class_labels = lb.classes_  # Get class names

    # Initialize variables for weighted ROC
    weighted_fpr = np.linspace(0, 1, 100)  # Interpolation points
    weighted_tpr = np.zeros_like(weighted_fpr)
    class_counts = np.sum(y_onehot_true, axis=0)  # Class distribution

    # Compute ROC for each class (OvR approach)
    plt.figure(figsize=(7, 5))
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_onehot_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        class_weight = class_counts[i] / len(y_train)  # Weight by class prevalence

        # Interpolate TPR at common FPR points
        interp_tpr = np.interp(weighted_fpr, fpr, tpr)
        weighted_tpr += interp_tpr * class_weight  # Weighted sum

        # Plot individual class ROC curves
        plt.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.3f})")

    # Plot the final weighted ROC curve
    plt.plot(weighted_fpr, weighted_tpr, color='black', linestyle="--", label="Weighted ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Weighted ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.show()

    # Overall Weighted AUC
    macro_auc = roc_auc_score(y_onehot_true, y_score, average="macro", multi_class="ovr")
    weighted_auc = roc_auc_score(y_onehot_true, y_score, average="weighted", multi_class="ovr")
    print(f"Macro-Averaged AUC: {macro_auc:.3f}")
    print(f"Weighted-Averaged AUC: {weighted_auc:.3f}")

