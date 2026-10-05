import Orange
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Orange.evaluation import CrossValidation
from Orange.evaluation.testing import LeaveOneOut
from Orange.classification import SVMLearner


raw_data = pd.read_excel("AllScores.xlsx", sheet_name="ZScores")


# Process the data into features and target (as we discussed earlier)
def filter_data(data):
    firstThreeChars = data["FirstThreeChars"]
    ZScores = data.iloc[:,2:]
    features = ZScores.loc[:, ['bFDRP_SLO_gis001', 'ADRP_SLO__gis001', 'CJDRP_gis001_004']]

    filtered_data = pd.DataFrame()
    filtered_data["FirstThreeChars"] = firstThreeChars  # Add the firstThreeChars column
    filtered_data = pd.concat([filtered_data, features], axis=1)  # Add the selected features

    '''Choose patients for training and testing'''

    train_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['AD_', 'CJD', 'FTD', 'NC-'])]
    test_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['swd'])]

    targets = train_data["FirstThreeChars"]

    # Initialize the label encoder
    label_encoder = LabelEncoder()
    encode_targets = label_encoder.fit_transform(targets)

    train_data["targets"] = encode_targets



    print(train_data)

def filter_data2(data):
    '''Filter relevant features and separate training/testing data.'''

    # Select the target variable and features
    firstThreeChars = data["FirstThreeChars"]
    ZScores = data.iloc[:, 2:]
    features = ZScores.loc[:, ['bFDRP_SLO_gis001', 'ADRP_SLO__gis001', 'CJDRP_gis001_004']]

    # Filtered data with features
    filtered_data = pd.DataFrame()
    filtered_data["FirstThreeChars"] = firstThreeChars  # Add the target column
    filtered_data = pd.concat([filtered_data, features], axis=1)  # Add the selected features

    # Choose training data
    train_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['AD_', 'CJD', 'FTD', 'NC-'])]
    test_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['swd'])]

    # Encode targets
    targets = train_data["FirstThreeChars"]
    label_encoder = LabelEncoder()
    encoded_targets = label_encoder.fit_transform(targets)

    # Add the encoded target column (but we will drop this in the Orange Table)
    train_data["encoded_targets"] = encoded_targets

    # Remove the first column (i.e., the original target column 'FirstThreeChars')
    train_data = train_data.drop(columns=["FirstThreeChars"])

    # Define the domain (features are continuous, target is discrete)
    target_classes = list(label_encoder.classes_)  # Original target classes
    domain = Orange.data.Domain(
        [Orange.data.ContinuousVariable(name) for name in features.columns],  # Features
        Orange.data.DiscreteVariable("FirstThreeChars", values=target_classes)  # Encoded target
    )

    # Convert to Orange Table
    orange_train_data = Orange.data.Table(domain, train_data[features.columns].values, encoded_targets)

    # Return the Orange Table for further use
    return orange_train_data, test_data



# Filter the data
train_data, test_data = filter_data2(raw_data)

# Initialize the SVM model
svm = SVMLearner(kernel="linear")  # You can change to other kernels like 'rbf' or 'poly'

# Apply Leave-One-Out Cross Validation
results = Orange.evaluation.testing.LeaveOneOut(store_data=train_data, store_models=svm)

accuracy = Orange.evaluation.CA(results)