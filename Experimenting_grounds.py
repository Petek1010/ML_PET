import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
import pandas as pd


def compute_roc_for_class(y_train ,y_score, class_of_interest):
    # Binarize labels for one-vs-rest classification
    classes = np.unique(y_train)
    y_onehot = label_binarize(y_train, classes=classes)


    # Find class index
    if class_of_interest not in classes:
        raise ValueError(f"Class {class_of_interest} not found in dataset.")

    class_id = np.flatnonzero(classes == class_of_interest)[0]

    # Compute ROC curve and AUC for the selected class
    fpr, tpr, _ = roc_curve(y_onehot[:, class_id], y_score[:, class_id])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance Level")

    # Formatting
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {class_of_interest} vs Rest")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print(f"AUC for {class_of_interest}: {roc_auc:.3f}")

# Example usage:
# compute_roc_for_class(your_dataframe, "AD_")
