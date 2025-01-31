from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def SVM(train_data, test_data):

    # Setting train data on features and targets
    X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
    y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

    # Model and validation setup
    svm_model = SVC(kernel='linear', C=1.0, gamma='scale')
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
        svm_model.fit(X_loo_train, y_loo_train)

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