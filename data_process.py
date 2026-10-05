import sys

import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer


class DataProcess:
    """
        Process raw data from a file, handle missing values, normalization, and encoding.
    """
    def __init__(self, input_targets, input_features, data_file: str, sheet: str='', country : str = 'SI'):

        self.feature_SI_dic = {
            "ADRP": ["ADRP_SLO__gis001"],
            "CBDRP": ["CBDRP_SLOV_gis001"],
            "CJDRP": ["CJDRP_gis001_004"],
            "DLBRP": ["DLBRP_SLO_spm5_gis001"],
            "BFDRP": ["bFDRP_SLO_gis001"],
            "MSARP": ["MSARP_SLO_gis001"],
            "PDRP": ["PDRP_SLO_gis001"],
            "PSPRP": ["PSPRP_SLO_gis001"],
            "DMN": ["DMN_NY__gis001"],  # Samo ameriški?
            #"HDRP": ["HDRP_gis001"],
            "PDCP": ["PDCP_NY_gis002"]  # Samo ameriški?
        }

        self.feature_USA_dic = {
            "ADRP": ["ADRP_NY_gis002"],
            "CBDRP": ["CBDRP1010_NY_gis001"],
            "CJDRP": ["CJDRP_gis001_004"],  # Ali je samo slovenski?
            "DLBRP": ["DLBRP_SLO_spm5_gis001"],  # Samo slovenski...
            "BFDRP": ["bFDRP_NY_gis001"],
            "MSARP": ["MSA_NY_gis001"],
            "PDRP": ["PDRP_NY_gis001"],
            "PSPRP": ["PSPRP_NY_gis001"],
            "DMN": ["DMN_NY__gis001"],  # Samo ameriški?
            "HDRP": ["HDRP_gis001"]  # samo slovenski?
        }

        self.target_dic = {
            "AD": ["AD_"],
            "CBD": ["CBD"],
            "CJD": ["CJD"],
            "DLB" : ["DLB"],
            "FTD": ["FTD"],
            "MSA": ["MSA"],
            "NC": ["NC-"],
            "PD": ["PD-"],
            "PSP": ["PSP"]
        }

        # Prevalence data (per 100,000):
        self.prior_data_dic = {
            "AD_": 1500,
            "CBD": 3,
            "CJD": 0.15,
            "DLB" : 150,
            "FTD": 15,
            "MSA": 4,
            "PD-": 300,
            "PSP": 5,
            "NC-": 0
        }

        self.population_distribution = {
            'AD_': 1500 / 100000,
            'DLB': 150 / 100000,
            'FTD': 15 / 100000,
            'CJD': 0.15 / 100000,
            'PD-': 300 / 100000,
            'MSA': 4 / 100000,
            'PSP': 5 / 100000,
            'CBD': 3 / 100000,
            'NC-': 0
        }

        self._check_user_input(input_targets, input_features)
        self.file = self._read_file(data_file, sheet)
        self.raw_data = self._handle_missing_data(self.file)
        self.targets, self.features, self.all_features = self._process_user_input(input_targets, input_features, country)

        self.le = LabelEncoder()
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.priors = None
        self.population_priors = None

    def _check_user_input(self, input_targs, input_feats):
        """ Verify that all provided target diseases and feature markers exist in known dictionaries. """
        for disease in input_targs:
            disease = disease.strip().upper()
            if disease not in self.target_dic:
                raise ValueError(f"Invalid target disease provided: '{disease}', check clinical_question_roster")

        for marker in input_feats:
            marker = marker.strip().upper()
            if marker not in self.feature_SI_dic:
                raise ValueError(f"Invalid feature marker provided: '{marker}', check clinical_question_roster")

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
        """ Fill missing values in the dataset using the mean of each column. """
        imputer = SimpleImputer(strategy='mean')
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def _process_user_input(self, input_targets, input_features, country):
        """ Map user-provided diseases and features to clinical codes based on the selected country. """
        clinical_targets = [self.target_dic[disease][0] for disease in input_targets if disease in self.target_dic]
        clinical_features = None
        all_features = None

        if country == "SI":
            clinical_features = [self.feature_SI_dic[marker][0] for marker in input_features if
                                 marker in self.feature_SI_dic]
            all_features = [v[0] for v in self.feature_SI_dic.values()]
        elif country == "USA":
            clinical_features = [self.feature_USA_dic[marker][0] for marker in input_features if
                                 marker in self.feature_USA_dic]
            all_features = [v[0] for v in self.feature_USA_dic.values()]

        print("Targets: ", clinical_targets)
        print("Features: ", clinical_features)

        return clinical_targets, clinical_features, all_features

    def split_data(self, data:pd.DataFrame, train_targets=None, test_targets=None, encode: bool = False ):
        """ Splits the dataset into training and testing sets based on target categories. """
        if train_targets is None:
            raise ValueError("train_targets must be provided.")
        if test_targets is None:
            raise ValueError("test_targets must be provided.")

        train_data = data.loc[data['FirstThreeChars'].isin(train_targets)]
        test_data = data.loc[data['FirstThreeChars'].isin(test_targets)]

        # Encode like ["AD", "FTD", "PD"] → [0, 1, 2]
        if encode:
            self.y_train = self.le.fit_transform(train_data["FirstThreeChars"])
            # Do NOT encode test labels – we don’t have ground truth for them
        else:
            self.y_train = train_data["FirstThreeChars"]
            self.y_test = test_data["FirstThreeChars"].values

        return train_data, test_data

    def normalize_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, scaler_type: str='standard'):
        """ Applies normalization using different scalers. It does not separate by target when scaling.
        It treats all training data together as one dataset, regardless of the target class. """
        scalers = {
            'standard': StandardScaler(),
            'minMax': MinMaxScaler(feature_range=(-1, 1)),
            'maxAbs': MaxAbsScaler(),
            'robust': RobustScaler()
        }
        scaler = scalers.get(scaler_type)
        # Select only numeric columns for normalization
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns

        train_data = train_data.copy()
        test_data = test_data.copy()

        train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols].values)
        test_data[numeric_cols] = scaler.transform(test_data[numeric_cols].values)

        return train_data, test_data

    def discretize_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, n_bins: int = 10):
        """ Bin continuous data into intervals. """
        # CHECK IF ITS DOING WHAT IS SUPPOSED TO DO - Is not in use rn
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns

        train_data = train_data.copy()
        test_data = test_data.copy()

        train_data[numeric_cols] = discretizer.fit_transform(train_data[numeric_cols].values)
        test_data[numeric_cols] = discretizer.transform(test_data[numeric_cols].values)

        return train_data, test_data

    def set_features(self, train_data: pd.DataFrame, test_data: pd.DataFrame, clinical_features=None):
        """ Extracts features based on predefined or user-specified clinical markers. """
        self.X_train = train_data[clinical_features]
        self.X_test = test_data[clinical_features]

    def set_priors(self, y_train):
        present_classes, counts = np.unique(y_train, return_counts=True)
        values = []
        for cls in present_classes:
            if cls not in self.prior_data_dic:
                raise ValueError(f"Missing prior for class {cls}")
            values.append(self.prior_data_dic[cls])

        # Normalize priors
        values = np.array(values, dtype=float)
        priors = values / values.sum()

        self.priors = priors

    def set_population_prioirs(self, y_train):
        present_classes, counts = np.unique(y_train, return_counts=True)

        priors_dict = {}
        for cls in present_classes:
            if cls not in self.population_distribution:
                raise ValueError(f"Missing prior for class {cls}")
            priors_dict[cls] = self.population_distribution[cls]

        # Build dict with only the present classes
        priors_dict = {cls: self.population_distribution[cls] for cls in present_classes}

        # Normalize values so they sum to 1
        total = sum(priors_dict.values())
        priors_dict = {cls: val / total for cls, val in priors_dict.items()}

        self.population_priors = priors_dict

    def check_feature_distribution(self, data: pd.DataFrame, plot: bool, max_plots: int = 16, alpha: float = 0.05):
        """ Check if features fall into normal distribution """
        if not plot:
             return

        n_features = len(self.all_features)
        plots_per_page = min(max_plots, n_features)
        n_cols = int(math.sqrt(plots_per_page))
        n_rows = math.ceil(plots_per_page / n_cols)
        results = []

        for i, feat in enumerate(self.all_features):
            values = data[feat].dropna()

            # Shapiro-Wilk test
            stat, p = stats.shapiro(values)
            normal = p > alpha
            results.append((feat, stat, p, normal))

            # Q-Q plot
            if i % plots_per_page == 0:
                if i > 0:
                    plt.tight_layout()
                    plt.show()
                plt.figure(figsize=(n_cols * 4, n_rows * 4))

            plt.subplot(n_rows, n_cols, (i % plots_per_page) + 1)
            stats.probplot(values, dist="norm", plot=plt)
            plt.title(f"{feat}\nShapiro p={p:.3g}, normal={normal}")

        plt.tight_layout()
        plt.show()


        # For individual plot - Can delete later
        individual = 1
        if individual:
            for feat in self.all_features:
                values = data[feat].dropna()

                # Shapiro-Wilk test
                stat, p = stats.shapiro(values)
                normal = p > alpha
                results.append((feat, stat, p, normal))

                # Q-Q plot for each feature individually
                plt.figure(figsize=(6, 6))
                stats.probplot(values, dist="norm", plot=plt)
                feat_key = next((k for k, v in self.feature_SI_dic.items() if feat in v), feat)
                plt.title(f"{feat_key}\nShapiro: p = {p:.3g}, normal = {normal}", fontsize=18)
                plt.xlabel("Teoretični kvantili", fontsize=14)
                plt.ylabel("Empirični kvantil", fontsize=14)
                plt.grid()
                plt.tight_layout()
                plt.show()  # each plot is shown individually


        # Return summary table
        return pd.DataFrame(results, columns=["Feature", "W-stat", "p-value", "Normal?"])

    def check_correlations(self, data: pd.DataFrame, plot: bool = False, threshold: float = 0.7):
        """ Pearson test on all features  """
        corr_matrix = data[self.all_features].corr(method='pearson')
        reverse_dic = {v[0]: k for k, v in self.feature_SI_dic.items()}


        if plot:
            corr_matrix = corr_matrix.rename(index=reverse_dic, columns=reverse_dic)
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",         # choose colormap
                center=0,              # ensures 0 is white/neutral
                cbar_kws={'label': 'Pearson correlation'}
            )
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)  # tick font size
            cbar.set_label('Moč korelacije r', fontsize=18, rotation=270, labelpad=20)
            plt.title("Pearsonova korelacija metaboličnih vzorcev", fontsize=18)
            plt.savefig("Korelacija.png", dpi=600)
            plt.show()

        # Extract upper triangle (excluding diagonal)
        upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

        # Absolute correlations only
        abs_corr = upper.abs().stack()

        # Mean correlation across all feature pairs
        mean_corr = abs_corr.mean()

        # Max correlation
        max_corr = abs_corr.max()

        # Heuristic: decide if correlation is "low" or "high"
        if mean_corr > threshold:
            level = "high"
        else:
            level = "low"

        return {
            "mean_correlation": mean_corr,
            "max_correlation": max_corr,
            "level": level
        }

    def check_correlations_ABS(
            self,
            data: pd.DataFrame,
            plot: bool = False,
            threshold: float = 0.7,
            absolute_plot: bool = False
    ):
        """ Pearson test on all features """

        corr_matrix = data[self.all_features].corr(method='pearson')
        reverse_dic = {v[0]: k for k, v in self.feature_SI_dic.items()}

        # --- PLOT ---
        if plot:
            plot_matrix = corr_matrix.abs() if absolute_plot else corr_matrix
            plot_matrix = plot_matrix.rename(index=reverse_dic, columns=reverse_dic)

            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                plot_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r" if not absolute_plot else "Reds",
                center=0 if not absolute_plot else None,
                cbar_kws={'label': 'Pearson correlation'}
            )

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(
                'Moč korelacije |r|' if absolute_plot else 'Moč korelacije r',
                fontsize=18,
                rotation=270,
                labelpad=20
            )

            title = "Pearsonova korelacija metaboličnih vzorcev"
            if absolute_plot:
                title += " "

            plt.title(title, fontsize=18)
            plt.savefig("KorelacijaABS.png", dpi=600)
            plt.show()

        # --- STATISTICS ---
        upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
        abs_corr = upper.abs().stack()

        mean_corr = abs_corr.mean()
        max_corr = abs_corr.max()

        level = "high" if mean_corr > threshold else "low"

        return {
            "mean_correlation": mean_corr,
            "max_correlation": max_corr,
            "level": level
        }


    def get_population_distribution(self):
        return pd.Series(self.population_distribution)

    def get_targets(self):
        return self.y_train, self.y_test

    def get_features(self):
        return self.X_train, self.X_test

    def get_priors(self):
        return self.priors

    def get_population_priors(self):
        return self.population_priors

    def get_raw_data(self):
        return self.file

    def get_decoded_labels(self):
        return self.le.inverse_transform(self.y_train)

    def get_clinical_targets(self):
        if self.targets is None:
            raise ValueError("Clinical targets have not been set.")
        return self.targets

    def get_class_distribution(self):
        print("Class distribution:")
        print(self.y_train.value_counts())
        return self.y_train.value_counts()


class GeneralPreprocess(DataProcess):

    """ Preprocess for general model"""
    def pipeline(self):
        train_data, test_data = self.split_data(self.raw_data, train_targets=self.targets, test_targets=['swd'], encode=False)
        train_data, test_data = self.normalize_data(train_data, test_data)
        self.set_features(train_data, test_data, clinical_features=self.features)


        self.check_feature_distribution(train_data, plot=False)
        self.check_correlations_ABS(train_data, plot=True, absolute_plot=True)
        sys.exit()
        X_train, X_test = self.get_features()
        y_train, y_test = self.get_targets()

        self.set_priors(y_train)
        self.set_population_prioirs(y_train) # V testiranju
        print('---------- Data preprocess DONE ---------')
        return X_train, y_train, X_test, y_test



