from statistics import LinearRegression

import numpy as np
import pandas as pd
import visualize as vis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

def logistic_regression(train_data, test_data):
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data["FirstThreeChars"].values

    X_test = test_data.iloc[:, 1:].values

    # Standardize the features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    strength = 0.8 # For the same result as Orange set C=4 here and C=1 in Orange

    log_reg = LogisticRegression(penalty='l2', C=strength ,solver='lbfgs', random_state=42)
    loo = LeaveOneOut()

    # Perform predictions using LOO cross-validation
    y_pred = cross_val_predict(log_reg, X_train_norm, y_train, cv=loo)


    # Compute classification report
    print("Classification Report:")
    print(classification_report(y_train, y_pred, digits=3))

    # Confusion Matrix

    vis.conf_matrix(y_train, y_pred, vis=True)

    # Compute accuracy score using cross-validation
    accuracy = cross_val_score(log_reg, X_train_norm, y_train, cv=loo).mean()
    print(f"LOO Accuracy: {accuracy:.3f}")




def grid_search(train_data):
    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    # Normalize and scale train and test features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    C_options = []
    for i in range(1,10,1):
        C_options.append(i/100)
        C_options.append(i/10)
        C_options.append(i)
        C_options.append(i*10)
    C_options.sort()
    print(C_options)




    # Hyperparameter grid
    param_grid_saga = {
        'C': [0.01, 0.07 ,0.1, 0.4, 0.6, 0.9, 1, 4, 6, 10, 100],  # Regularization strength (inverse)
        'penalty': ['l1', 'l2'],  # L2 regularization for saga
    }

    param_grid = {
        'C': C_options,  # Regularization strength (inverse)
        'penalty': ['l2'],  # L2 regularization for lbfgs
    }

    loo = LeaveOneOut()
    log_reg = LogisticRegression(solver='lbfgs',max_iter=10000 ,random_state=42)

    # GridSearchCV with LOO
    grid_search = GridSearchCV(log_reg, param_grid, cv=loo, scoring='accuracy')
    grid_search.fit(X_train_norm, y_train)

    # Print the best parameters
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)


