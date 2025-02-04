import sys

import seaborn as sns
import visualize as vis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from sklearn.svm import SVC
from sklearn import datasets, svm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay



def SVM(train_data, test_data):

    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    # Model and validation setup
    svm_model = SVC(kernel='linear', C=1.0)
    loo = LeaveOneOut()

    ''' model validation '''
    # Store predictions and true values
    y_true_loo = []
    y_pred_loo = []

    # Perform leave one out cross-validation
    for train_index, val_index in loo.split(X_train):
        # Split the training data into train and validation sets
        X_loo_train, X_loo_val = X_train[train_index], X_train[val_index]
        y_loo_train, y_loo_val = y_train[train_index], y_train[val_index]

        # Train the SVM model on the training subset
        histroy = svm_model.fit(X_loo_train, y_loo_train)

        # Validation
        y_pred_loo.append(svm_model.predict(X_loo_val)[0]) # predicted value
        y_true_loo.append(y_loo_val[0]) # validation target == true value




    # Evaluate the LOO performance
    loo_accuracy = accuracy_score(y_true_loo, y_pred_loo)
    print("Leave-One-Out Accuracy on Training Data:", loo_accuracy)

    # Classification report for precision, recall, and F1-score
    print("\nClassification Report:")
    print(classification_report(y_true_loo, y_pred_loo, target_names=sorted(set(y_train))))

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_loo, y_pred_loo, labels=sorted(set(y_train)))
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Calculate specificity manually from confusion matrix
    specificity_per_class = {}
    for i, label in enumerate(sorted(set(y_train))):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class[label] = specificity

    print("\nSpecificity per class:")
    for label, spec in specificity_per_class.items():
        print(f"Class {label}: {spec:.2f}")


    ''' Train on Full Training Data and Predict on Real Test Data '''
    print("\nTraining on Full Dataset and Predicting on Test Data...")

    # Train the model on the full training dataset
    svm_model.fit(X_train, y_train)

    # Extract features from test data (excluding the target column if present)
    X_test = test_data.iloc[:, 1:].values  # Use same features as training data

    # Make predictions
    test_predictions = svm_model.predict(X_test)

    # Add predictions to the test dataframe
    test_data["Predicted_Class"] = test_predictions

    # Display the first few rows with predictions
    print("\nPredictions on Test Data:")
    print(test_data.head())

    return test_data

def cross_val_svm(train_data, test_data):
    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    X_test = test_data.iloc[:, 1:].values

    # Normalized and scaled data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train) # Apply also for testing data !!!
    X_test_norm = scaler.fit_transform(X_test)

    # Model and validation setup
    svm_model = SVC(kernel='linear', C=1.0, decision_function_shape="ovr", random_state=42)
    #svm_model = svm.LinearSVC(C=1.0)
    loo = LeaveOneOut()

    # Predicted values
    y_pred = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo)

    # Classification report
    print('\n Classification report: \n',classification_report(y_train, y_pred, digits=3))

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    # print("Confusion Matrix:\n", cm)
    labels = np.unique(y_train)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Greens")
    ax.set_title("Confusion Matrix")
    plt.show()


    # Compute recall (sensitivity) per class [TP/[TP+FN]]
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    labels = np.unique(y_train)
    print("\nRecall for each class (Per-Class Accuracy):")
    for label, acc in zip(labels, per_class_accuracy):
        print(f"Class {label}: {acc:.2f}")


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


















def PCA(X_train, y_train):
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    # Plot PCA-transformed data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="viridis", alpha=0.7)
    plt.title("PCA Projection of Training Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Classes")
    plt.show()