import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets



def importData(fileName, sheetName = ''):
    '''Import all patients'''
    # TO DO: import data can read csv or excel file
    readData = pd.read_excel(fileName, sheet_name=sheetName)

    #print(readData.head())
    return readData


def filterData(data):

    firstThreeChars = data["FirstThreeChars"]
    ZScores = data.iloc[:,2:]
    features = ZScores.loc[:, ['bFDRP_SLO_gis001', 'ADRP_SLO__gis001', 'CJDRP_gis001_004']]

    filtered_data = pd.DataFrame()
    filtered_data["FirstThreeChars"] = firstThreeChars  # Add the firstThreeChars column
    filtered_data = pd.concat([filtered_data, features], axis=1)  # Add the selected features

    '''Choose patients for training and testing'''

    train_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['AD_', 'CJD', 'FTD', 'NC-'])]
    test_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['swd'])]

    print(train_data)

    return train_data, test_data


def SVM(train_data, test_data):

    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    # Model and validation setup
    svm_model = SVC(kernel='linear', C=1.0, gamma='scale')
    loo = LeaveOneOut()

    # Store predictions and true values
    y_true_loo = []
    y_pred_loo = []

    # Perform leave one out cross-validation
    for train_index, val_index in loo.split(X_train):
        # Split the training data into train and validation sets
        X_loo_train, X_loo_val = X_train[train_index], X_train[val_index]
        y_loo_train, y_loo_val = y_train[train_index], y_train[val_index]

        # Train the SVM model on the training subset
        svm_model.fit(X_loo_train, y_loo_train)

        # Validation
        y_pred_loo.append(svm_model.predict(X_loo_val)[0])
        y_true_loo.append(y_loo_val[0])

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






def simple_svm(train_data):

    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    # Model and validation setup
    svm_model = SVC(kernel='linear', C=1.0, gamma='scale')
    loo = LeaveOneOut()

    # Store predictions and true values
    accuracies = []

    # Perform leave one out cross-validation
    for train_index, val_index in loo.split(X_train):
        # Split the training data into train and validation sets
        X_loo_train, X_loo_val = X_train[train_index], X_train[val_index]
        y_loo_train, y_loo_val = y_train[train_index], y_train[val_index]

        # Train the SVM model on the training subset
        svm_model.fit(X_loo_train, y_loo_train)

        y_pred = svm_model.predict(X_loo_val)

        # Validation
        accuracies.append(accuracy_score(y_loo_val, y_pred))

    # Calculate overall accuracy
    mean_accuracy = np.mean(accuracies)



    print(f"Mean accuracy (LOOCV): {mean_accuracy}")




if __name__ == '__main__':
    print("Start program")

    rawData = importData("AllScores.xlsx", "ZScores")
    train_data, test_data = filterData(rawData)
    #SVM(train_data, test_data)
    simple_svm(train_data)
