import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def logistic_regression(train_data, test_data):
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data["FirstThreeChars"].values

    X_test = test_data.iloc[:, 1:].values

    # Standardize the features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    # Define logistic regression model
    log_reg = LogisticRegression(penalty='l2', C=1.0,solver='liblinear', random_state=42)


    # Set up Leave-One-Out cross-validation
    loo = LeaveOneOut()

    # Perform predictions using LOO cross-validation
    y_pred = cross_val_predict(log_reg, X_train_norm, y_train, cv=loo)

    # Compute classification report
    print("Classification Report:")
    print(classification_report(y_train, y_pred, digits=3))

    # Compute accuracy score using cross-validation
    accuracy = cross_val_score(log_reg, X_train_norm, y_train, cv=loo).mean()
    print(f"LOO Accuracy: {accuracy:.3f}")

