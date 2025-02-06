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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc



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



def cross_val_svm(train_data, test_data):
    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    X_test = test_data.iloc[:, 1:].values

    # Normalize and scale train and test features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Model and validation setup
    svm_model = SVC(kernel='linear', C=1.0, decision_function_shape="ovr", random_state=42, probability=True)
    loo = LeaveOneOut()

    # Predicted values
    y_pred = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo)

    # Classification report
    print('\n Classification report: \n',classification_report(y_train, y_pred, digits=3))

    # Create Confusion matrix
    cm = vis.conf_matrix(y_train,y_pred,0)
    vis.recall(cm, y_train, 0)
    vis.classification_accuracy(cm, y_train)








    # Print overall accuracy (score):
    scores = cross_val_score(svm_model, X_train_norm, y_train, cv=loo)
    print('\nModel overall accuracy:')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


    ''' Lean on test data and perform predictions '''
    svm_model.fit(X_train_norm,y_train)

    y_pred_test = svm_model.predict(X_test_norm)

    test_data["PredictedClass"] = y_pred_test
    print('\n--------------- PREDICTIONS ------------')
    print("\nPredictions on Test Data:")
    print(test_data.head())


    ''' ROC analysis '''
    # Convert y_train (true labels) to one-hot encoding
    lb = LabelBinarizer()
    y_onehot_true = lb.fit_transform(y_train)
    y_score = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo, method="predict_proba") # <-- the best so far




    # Overall Weighted AUC
    macro_auc = roc_auc_score(y_onehot_true, y_score, average="macro", multi_class="ovr")
    weighted_auc = roc_auc_score(y_onehot_true, y_score, average="weighted", multi_class="ovr")
    print(f"Macro-Averaged AUC: {macro_auc:.3f}")
    print(f"Weighted-Averaged AUC: {weighted_auc:.3f}")

    #plot_weighted_roc(svm_model,X_train_norm,y_train)


    # Select class of interest for one-vs-rest ROC
    class_of_interest = "AD_"
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
    _=display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"One-vs-Rest ROC Curve:\n{class_of_interest} vs All",
    )
    plt.show()





    ''' AUC Average over all classes '''
    # Initialize a list to store AUC scores for each class
    auc_scores = []

    # Iterate over all classes and calculate AUC
    for class_id in range(y_onehot_true.shape[1]):  # Iterate over all classes
        auc = roc_auc_score(y_onehot_true[:, class_id], y_score[:, class_id])
        auc_scores.append(auc)
        print(f"AUC for class {lb.classes_[class_id]}: {auc:.3f}")

    # Calculate the average AUC across all classes
    average_auc = np.mean(auc_scores)
    print(f"Average AUC across all classes: {average_auc:.3f}")

    # Now plot ROC curves for each class
    ''' for class_id in range(y_onehot_true.shape[1]):
        display = RocCurveDisplay.from_predictions(
            y_onehot_true[:, class_id],  # True labels (one-hot)
            y_score[:, class_id],  # Model scores (probabilities)
            name=f"{lb.classes_[class_id]} vs the rest",
            color="darkorange",
            plot_chance_level=True,
            despine=True,
        )
        display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"One-vs-Rest ROC Curve: {lb.classes_[class_id]} vs All",
        )
        plt.show()'''


    ###########################################################










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


def plot_weighted_roc(svm_model, X_train_norm, y_train):
    # Use Leave-One-Out cross-validation for probability estimates
    loo = LeaveOneOut()
    y_score = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo, method="predict_proba")

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
        plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {roc_auc:.3f})")

    # Plot the final weighted ROC curve
    plt.plot(weighted_fpr, weighted_tpr, color='black', linestyle="--", label="Weighted ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Weighted ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.show()