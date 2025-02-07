import sys

import sklearn.metrics


import visualize as vis
import Experimenting_grounds as ex

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from sklearn.svm import SVC
from sklearn import datasets, svm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
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



def cross_val_svm_default(train_data, test_data):
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

    # Predicted values and probability (svm_model probability = True) scores
    y_pred = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo)
    y_score = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo, method="predict_proba")

    # Classification report
    print('\n Classification report: \n',classification_report(y_train, y_pred, digits=3))

    # Confusion matrix, recall and classification accuracy
    cm = vis.conf_matrix(y_train,y_pred,0)
    vis.recall(cm, y_train, 0)
    vis.classification_accuracy(cm, y_train)

    # ROC analysis (Not the same as ORANGE !!!)
    #vis.ROC_analysis_target(y_score,y_train,'AD_')
    #vis.ROC_analysis_all_targets(y_score, y_train)

    # Print overall accuracy (score):
    scores = cross_val_score(svm_model, X_train_norm, y_train, cv=loo)
    print('--------------------------------------------------------')
    print('\nModel overall accuracy:')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


    ''' Lean on test data and perform predictions '''
    svm_model.fit(X_train_norm,y_train)

    y_pred_test = svm_model.predict(X_test_norm)

    test_data["PredictedClass"] = y_pred_test
    print('\n=============== PREDICTIONS ==========================')
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

    # Predicted values and probability (svm_model probability = True) scores
    y_pred = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo)
    y_score = cross_val_predict(svm_model, X_train_norm, y_train, cv=loo, method="predict_proba")

    # Classification report
    print('\n Classification report: \n',classification_report(y_train, y_pred, digits=3))

    # Confusion matrix, recall and classification accuracy
    cm = vis.conf_matrix(y_train,y_pred,0)
    vis.recall(cm, y_train, 0)
    vis.classification_accuracy(cm, y_train)

    # ROC analysis (Not the same as ORANGE !!!)
    #vis.ROC_analysis_target(y_score,y_train,'AD_')
    #vis.ROC_analysis_all_targets(y_score, y_train)

    # Print overall accuracy (score):
    scores = cross_val_score(svm_model, X_train_norm, y_train, cv=loo)
    print('--------------------------------------------------------')
    print('\nModel overall accuracy:')
    print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))


    ''' Lean on test data and perform predictions '''
    svm_model.fit(X_train_norm,y_train)

    y_pred_test = svm_model.predict(X_test_norm)

    test_data["PredictedClass"] = y_pred_test
    print('\n=============== PREDICTIONS ==========================')
    print("\nPredictions on Test Data:")
    print(test_data.head())



def grid_search(train_data):
    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    # Normalize and scale train and test features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)


    # Grid Search
    param_grid = {
        'C': [0.1, 0.4, 0.8, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly','sigmoid'],
        'degree':[1, 2, 3, 4, 5, 6, 7, 8],
        'gamma': ['scale', 'auto']
    }

    # Initialize SVM
    svm = SVC(probability=True, decision_function_shape="ovr")
    loo = LeaveOneOut()

    # Grid search with cross-validation (5-fold)
    grid_search = GridSearchCV(svm, param_grid, cv=loo, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_norm, y_train)

    # Print the best parameters
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)




















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

