import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer


class DataProcess:
    """
        Process raw data from a file, handle missing values, normalization, and encoding.
    """
    def __init__(self, data_file: str, sheet: str=''):
        self.file = self._read_file(data_file, sheet)
        self.raw_data = self._handle_missing_data(self.file)

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.disease_dic = {
            "AD" : ["ADRP_NY_gis002", "ADRP_SLO__gis001"],
            "CBD" : ["CBDRP1010_NY_gis001" , "CBDRP_SLOV_gis001"],
            "CJD" : ["CJDRP_gis001_004"],
            "DLB" : ["DLBRP_SLO_spm5_gis001"],
            "FTD" : ["bFDRP_NY_gis001", "bFDRP_SLO_gis001"],
            "MSA" : ["MSARP_SLO_gis001", "MSA_NY_gis001"],
            "PD" : ["PDRP_NY_gis001", "PDRP_SLO_gis001"],
            "PSP" : ["PSPRP_NY_gis001", "PSPRP_SLO_gis001"]
        }

    def _read_file(self, data_file: str, sheet: str):
        """ Reads an Excel or CSV file. """
        try:
            file = pd.read_excel(data_file, sheet_name=sheet)
            return file
        except Exception:
            try:
                file = pd.read_csv(data_file)
                return file
            except Exception as e:
                return ValueError(f"Unknown file type: {e}")

    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Checks for missing values and prompts the user to impute with mean if needed. """
        missing_count = data.isnull().sum().sum()

        if missing_count > 0:
            print(f'Missing values detected: {missing_count}')
            ans = input('Do you want to impute missing values with mean? (y/n) ').lower()
            if ans == "y":
                return self._fill_missing_data(data)
        return data

    def _fill_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        imputer = SimpleImputer(strategy='mean')
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def split_data(self, data:pd.DataFrame, train_targets=None, test_targets=None, encode: bool = True ):
        """ Splits the dataset into training and testing sets based on target categories. """
        if train_targets is None:
            train_targets = ['AD_', 'CJD', 'FTD', 'NC-']
        if test_targets is None:
            test_targets = ['swd']

        train_data = data.loc[data['FirstThreeChars'].isin(train_targets)]
        test_data = data.loc[data['FirstThreeChars'].isin(test_targets)]

        if encode:
            self.le = LabelEncoder()
            self.y_train = self.le.fit_transform(train_data["FirstThreeChars"])
        else:
            self.y_train = train_data["FirstThreeChars"]

        self.y_test = test_data["FirstThreeChars"].values

        return train_data, test_data

    def set_test_data_deprecated(self, data):
        # DEPRECATED
        test_targets = ['swd']
        test_data = data.loc[data.FirstThreeChars.isin(test_targets)]
        self.y_test = test_data["FirstThreeChars"].values
        return test_data

    def set_train_data_deprecated(self, data, clinical_input=None):
        # DEPRECATED
        train_targets = ['AD_', 'CJD', 'FTD', 'NC-']
        #train_targets = clinical_input
        train_data = data.loc[data.FirstThreeChars.isin(train_targets)]

        self.le = LabelEncoder()
        self.y_train = train_data["FirstThreeChars"].values
        #self.y_train = self.le.fit_transform(train_data["FirstThreeChars"].values)
        return train_data

    def normalize_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, scaler_type: str='standard'):
        """Applies normalization using different scalers."""
        scalers = {
            'standard': StandardScaler(),
            'minMax': MinMaxScaler(feature_range=(-1, 1)),
            'maxAbs': MaxAbsScaler(),
            'robust': RobustScaler()
        }
        scaler = scalers.get(scaler_type, StandardScaler())
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns

        train_data = train_data.copy()
        test_data = test_data.copy()

        train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols].values)
        test_data[numeric_cols] = scaler.transform(test_data[numeric_cols].values)

        return train_data, test_data

    def discretize_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, n_bins: int = 10):
        """ Bin continuous data into intervals. """
        # CHECK IF ITS DOING WHAT IS SUPPOSE TO DO
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns

        train_data = train_data.copy()
        test_data = test_data.copy()

        train_data[numeric_cols] = discretizer.fit_transform(train_data[numeric_cols].values)
        test_data[numeric_cols] = discretizer.transform(test_data[numeric_cols].values)

        return train_data, test_data

    def set_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame, clinical_features=None):
        """ Extracts features based on predefined or user-specified clinical markers. """
        if clinical_features is None:
            clinical_features = ['bFDRP_SLO_gis001', 'ADRP_SLO__gis001', 'CJDRP_gis001_004']

        self.X_train = train_data[clinical_features]
        self.X_test = test_data[clinical_features]

    def clinical_question(self):
        """Generates clinical features based on disease dictionary."""
        clinical_input = ['AD', 'CJD', 'FTD']
        clinical_features = [self.disease_dic.get(disease, [''])[0] for disease in clinical_input]

        clinical_input = ['AD_' if disease == 'AD' else disease for disease in clinical_input]
        clinical_input.append('NC-')

        return clinical_input, clinical_features

    def normalize_data_deprecated(self, train_data: pd.DataFrame, test_data: pd.DataFrame, scaler_type: str='standard'):
        """Normalize or scale data with different scalers. """
        scalers = {
            'standard': StandardScaler(),
            'minMax': MinMaxScaler(feature_range=(-1, 1)),
            'maxAbs': MaxAbsScaler(),
            'robust': RobustScaler()
        }
        scaler = scalers.get(scaler_type)

        # Identify numeric and non-numeric columns
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns

        # Separate numeric and non-numeric data
        train_non_numeric = train_data[non_numeric_cols]
        test_non_numeric = test_data[non_numeric_cols]

        train_features = train_data[numeric_cols]
        test_features = test_data[numeric_cols]

        # Normalize only numeric features
        train_features_norm = scaler.fit_transform(train_features)
        test_features_norm = scaler.transform(test_features)

        # Convert back to DataFrame
        train_features_norm_df = pd.DataFrame(train_features_norm, columns=numeric_cols, index=train_data.index)
        test_features_norm_df = pd.DataFrame(test_features_norm, columns=numeric_cols, index=test_data.index)

        # Concatenate non-numeric columns back
        train_data_norm = pd.concat([train_non_numeric, train_features_norm_df], axis=1)
        test_data_norm = pd.concat([test_non_numeric, test_features_norm_df], axis=1)

        return train_data_norm, test_data_norm



    def set_features_deprecated(self, train_data: pd.DataFrame, test_data: pd.DataFrame, clinical_features=None):
        features = ['bFDRP_SLO_gis001', 'ADRP_SLO__gis001', 'CJDRP_gis001_004']
        #features = clinical_features

        self.X_train = train_data.loc[:, features]
        self.X_test = test_data.loc[:, features]

    def discretize_data_deprecated(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

        # Identify numeric and non-numeric columns
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns

        # Separate numeric and non-numeric data
        train_non_numeric = train_data[non_numeric_cols]
        test_non_numeric = test_data[non_numeric_cols]

        train_features = train_data[numeric_cols]
        test_features = test_data[numeric_cols]

        # Discretize only numeric features
        train_features_disc = discretizer.fit_transform(train_features)
        test_features_disc = discretizer.transform(test_features)

        # Convert back to DataFrame
        train_features_disc_df = pd.DataFrame(train_features_disc, columns=numeric_cols, index=train_data.index)
        test_features_disc_df = pd.DataFrame(test_features_disc, columns=numeric_cols, index=test_data.index)

        # Concatenate non-numeric columns back
        train_data_disc = pd.concat([train_non_numeric, train_features_disc_df], axis=1)
        test_data_disc = pd.concat([test_non_numeric, test_features_disc_df], axis=1)

        return train_data_disc, test_data_disc


    def get_targets(self):
        return self.y_train, self.y_test

    def get_features(self):
        return self.X_train, self.X_test

    def get_raw_data(self):
        return self.file

    def clinical_question_deprecated(self):
        clinical_input = ['AD', 'CJD', 'FTD']
        clinical_features = []

        for i in range(len(clinical_input)):
            clinical_features.append(self.disease_dic.get(clinical_input[i])[0])
        print(clinical_features)
        # Correct clinical input
        for i in range(len(clinical_input)):
            if clinical_input[i] == 'AD':
                clinical_input[i] = 'AD_'
        clinical_input.append('NC-')

        return clinical_input, clinical_features

    def old_pipeline(self):
        clinical_input, clinical_features = self.clinical_question_deprecated()
        test_data = self.set_test_data_deprecated(self.raw_data)
        train_data = self.set_train_data_deprecated(self.raw_data,clinical_input)
        train_data, test_data = self.normalize_data_deprecated(train_data, test_data)
        self.set_features_deprecated(train_data,test_data)
        X_train, X_test = self.get_features()
        y_train, y_test = self.get_targets()

        return X_train, y_train, X_test, y_test


    def get_decoded_labels(self):
        return self.le.inverse_transform(self.y_train)


class SVMPreprocess(DataProcess):
    """ Preprocess for SVM model"""
    def preprocess(self):
        clinical_input, clinical_features = self.clinical_question()
        train_data, test_data = self.split_data(self.raw_data, clinical_input, encode=False)
        train_data, test_data = self.normalize_data(train_data, test_data)
        self.set_features(train_data, test_data)
        X_train, X_test = self.get_features()
        y_train, y_test = self.get_targets()

        return X_train, y_train, X_test, y_test
