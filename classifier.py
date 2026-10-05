import sys
from collections import Counter
from string import digits
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelBinarizer, KBinsDiscretizer, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    recall_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score
from tensorflow.python.ops.gen_tpu_ops import retrieve_tpu_embedding_frequency_estimator_parameters
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler


class Classifier:
    """
    Chose desired classifier and its hyperparameters. If no parameters are set, defaults will be used.

    Default parameters:

    - SVM: kernel='linear', C=1.0, decision_function_shape="ovr", random_state=42, probability=True

    - Logistic Regression: penalty='l2', C=4.0, solver='lbfgs', random_state=42

    - Naive Bayes: priors: None
    """
    def __init__(self, X_train, y_train, X_test, y_test,
                 class_priors = None,
                 bayes_priors = None,
                 classifier_name="svm",
                 **kwargs):

        self.classifiers = {
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "naive_bayes" : GaussianNB,
            "naive_bayes_priors" : GaussianNB,
            "naive_bayes_categorical": CategoricalNB,
            "random_forest": RandomForestClassifier,
            "gradient_boost": GradientBoostingClassifier,
            "knn": KNeighborsClassifier
        }

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classifier_name = classifier_name
        self.class_priors = class_priors
        self.bayes_priors = bayes_priors
        self.report_dict = None
        self.report_dict_corr = None

        self.auc_result_corr = None
        self.auc_result = None
        self.spec_corr = None


        self.classifier = self.classifiers[classifier_name](**kwargs) if kwargs else self._set_default_params()

    def _set_default_params(self):
        """ Sets default hyperparameters """
        class_counts = Counter(self.y_train)
        total_samples = len(self.y_train)
        class_prior = [class_counts[c] / total_samples for c in sorted(class_counts.keys())]

        default_params = {
            "svm": {"kernel": "linear", "C": 1, "decision_function_shape": "ovr", "class_weight" : "balanced" ,"random_state": 42, "probability": True},
            "logistic_regression": {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "class_weight" : "balanced", "random_state": 42, "max_iter":2000}, #c=4
            "naive_bayes" : {"priors":None, "var_smoothing":1e-9},
            "naive_bayes_priors": {"priors": self.bayes_priors, "var_smoothing": 1e-9},
            "naive_bayes_categorical": {"alpha": 1.0, "class_prior": class_prior},
            "random_forest": {"n_estimators":10, "criterion":"gini","min_samples_leaf":1, "min_samples_split":5, "max_features":"sqrt","bootstrap":True,"random_state": 5}
        }
        return self.classifiers[self.classifier_name](**default_params.get(self.classifier_name,{}))

    def set_best_parameters(self):
        """ Sets the best hyperparameters based on grid_parameter_search() """
        best_params = self.grid_parameter_search()
        if self.classifier_name == 'svm':
            self.classifier = SVC(**best_params, decision_function_shape="ovr", random_state=42, probability=True)
        elif self.classifier_name == 'logistic_regression':
            self.classifier = LogisticRegression(**best_params, solver='lbfgs', random_state=42)

    def manual_learning(self, model, X_train, y_train):
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        n = X.shape[0]

        preds = np.empty(n, dtype=object)  # object to allow strings / ints
        loo = LeaveOneOut()

        for train_idx, test_idx in loo.split(X):
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            preds[test_idx[0]] = m.predict(X[test_idx])[0]

        return preds

    def evaluate(self, cv=LeaveOneOut(), report = False, use_bayes_correction = False):
        """ Train model with Leave-one-out cross validation"""
        print("Evaluating:", self.classifier_name)
        # Discretizer for Naive Bayes CategoricalNB classification
        discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile') # EXACT ORANGE SETUP (n_bins=4, encode='ordinal', strategy='quantile')
        X_train_discrete = discretizer.fit_transform(self.X_train)

        if self.classifier_name == "svm":
            self.classifier = SVC(kernel='linear', C=1, decision_function_shape='ovr', random_state=42, class_weight='balanced', probability=True)
        elif self.classifier_name == "logistic_regression":
            self.classifier = LogisticRegression(max_iter=1000,penalty='l2',C=4,solver='lbfgs',class_weight='balanced',random_state=42)
        elif self.classifier_name == "naive_bayes":
            self.classifier = GaussianNB(priors=None)
        elif self.classifier_name == "naive_bayes_priors":
            self.classifier = GaussianNB(priors=self.bayes_priors)



        # Training model on imported data (returns predictions and probabilities)
        if self.classifier_name == 'naive_bayes_categorical':
            self.y_pred = cross_val_predict(self.classifier, X_train_discrete, self.y_train, cv=cv)
            self.y_probs = cross_val_predict(self.classifier, X_train_discrete, self.y_train, cv=cv, method="predict_proba")
        else:
            self.y_pred = cross_val_predict(self.classifier, self.X_train, self.y_train, cv=cv)
            self.y_probs = cross_val_predict(self.classifier, self.X_train, self.y_train, cv=cv, method="predict_proba")

        # Correcting y_probs with Bayes correction for general population (with priors)
        if use_bayes_correction:
            class_order = np.unique(self.y_train)
            if hasattr(self.classifier, 'classes_') and not np.array_equal(class_order, self.classifier.classes_):
                raise ValueError("Class order does not match classifier classes.")

            # Procentualna porazdelitev tarč v učnem setu [counting how many samples belong to each class]
            train_priors = np.array([np.mean(self.y_train == cls) for cls in class_order])
            # Procentualna porazdelitev tarč v populaciji 100 000
            population_priors = np.array([self.class_priors[cls] for cls in class_order])

            report_no_corr, report_corr, spec_corr1 = self.bayes_comparison(
                y_train=self.y_train,
                y_probs=self.y_probs,
                population_priors=population_priors,
                train_priors=train_priors,
                class_order=class_order
            )

            self.spec_corr = spec_corr1
            self.report_dict_corr = report_corr

            if report:
                print("---- WITHOUT Bayes Correction ----")
                print(report_no_corr)
                print("---- WITH Bayes Correction ----")
                print(report_corr)

        if report:
            print('\n Classification report for [', self.classifier_name,']: \n', classification_report(self.y_train, self.y_pred, digits=3))


        self.report_dict = classification_report(self.y_train, self.y_pred, digits=3, output_dict=True, zero_division=0)
        self.auc_result = self.one_vs_rest_ROC_AUC_all_targets(plot=False)
        #self.auc_result = roc_auc_score(self.y_train, self.y_probs, average="macro", multi_class="ovr")
        #print(self.auc_result)




    def bayes_correction(self, y_probs, population_priors, train_priors):
        adjustment_factors = population_priors / np.clip(train_priors, 1e-12, None)
        adjusted_probs = y_probs * adjustment_factors
        corrected_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True) # Normalization
        return corrected_probs

    def bayes_comparison(self, y_train, y_probs, population_priors, train_priors, class_order):
        # Without correction
        #y_pred = [class_order[i] for i in np.argmax(y_probs, axis=1)]
        report = classification_report(y_train, self.y_pred,
                                               target_names=class_order, digits=3, zero_division=0, output_dict=True)

        # With correction
        y_probs_corr = self.bayes_correction(y_probs, population_priors, train_priors)
        y_pred_corr = [class_order[i] for i in np.argmax(y_probs_corr, axis=1)]
        report_corr = classification_report(y_train, y_pred_corr,
                                            target_names=class_order, digits=3, zero_division=0, output_dict=True)
        self.auc_result_corr = self.one_vs_rest_ROC_AUC_all_targets(plot=False)

        # Specificity corrected: build confusion matrix with explicit labels order
        cm = confusion_matrix(y_train, y_pred_corr, labels=class_order)
        n_classes = cm.shape[0]

        specificities = []
        for i in range(n_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            specificities.append(spec)

        # average specificity across classes
        spec_corr = float(np.mean(specificities))


        return report, report_corr, spec_corr


    def get_model_evaluation(self):
        """ Get ORANGE params """
        self.classification_accuracy()
        recall = recall_score(self.y_train, self.y_pred, average='macro')  # or 'micro', 'weighted'
        print(f'Overall Recall (average): {recall:.3f}')

    def get_report_dictionary(self):
        return self.report_dict

    def get_corrected_report_dictionary(self):
        return self.report_dict_corr

    def get_params(self):
        return self.classifier.get_params()

    def get_spec_corr(self):
        return self.spec_corr

    def predict(self):
        """ Makes predictions on test data """
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)

    def grid_parameter_search(self, cv=LeaveOneOut()):
        """ Automatic search for best hyperparameters in range """
        C_params = sorted({i / 100 for i in range(1, 10)} |
                           {i / 10 for i in range(1, 10)} |
                           {i for i in range(1, 10)} |
                           {i * 10 for i in range(1, 10)})

        param_grid = {}
        if self.classifier_name == 'svm':
            param_grid = {
                'C': C_params,
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'degree': [1, 2, 3, 4],
                'gamma': ['scale', 'auto']
            }
        elif self.classifier_name == 'logistic_regression':
            param_grid = {
                'C': C_params,
                'penalty': ['l2']
            }

        grid_search = GridSearchCV(self.classifier, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_

        # Print the best parameters
        print("Best Parameters for [", self.classifier_name, "] are: ", best_params)
        print("Best Score:", grid_search.best_score_)
        return best_params

    def plot_learning_curve(self, scoring="f1_macro", cv=None, model_name=None):
        """
        Plots the learning curve for the current classifier.
        """
        if cv is None:
            cv = LeaveOneOut() #StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # default to LOO if nothing else is passed

        if model_name is None:
            model_name = type(self.classifier).__name__  # use class name of the model

        train_sizes, train_scores, test_scores = learning_curve(
            self.classifier,
            self.X_train, self.y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score=np.nan,
            train_sizes=np.linspace(0.5, 1.0, 10)
        )

        # Mean & std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Clip mean and shaded area to [0, 1]
        train_mean = np.clip(train_mean, 0, 1)
        test_mean = np.clip(test_mean, 0, 1)
        train_lower = np.clip(train_mean - train_std, 0, 1)
        train_upper = np.clip(train_mean + train_std, 0, 1)
        test_lower = np.clip(test_mean - test_std, 0, 1)
        test_upper = np.clip(test_mean + test_std, 0, 1)

        # Plot
        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_mean, label="Training score", color="blue")
        plt.fill_between(train_sizes, train_lower, train_upper, alpha=0.2, color="blue")

        plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green")
        plt.fill_between(train_sizes, test_lower, test_upper, alpha=0.2, color="green")

        plt.xlabel("Training set size")
        plt.ylabel(scoring)
        plt.title(f"Learning Curve: {model_name}")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_train, self.y_pred)
        labels = np.unique(self.y_train)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Greens")

        ax.set_title("Konfuzna matrika - " + str(self.classifier_name))
        plt.show()

    def specificity_score(self, average='macro'):
        cm = confusion_matrix(self.y_train, self.y_pred)
        n_classes = cm.shape[0]

        specificities = []
        for i in range(n_classes):
            # For class i
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            specificities.append(spec)

        if average == 'macro':
            return np.mean(specificities)
        elif average == 'weighted':
            class_counts = np.sum(cm, axis=1)
            return np.average(specificities, weights=class_counts)
        else:
            return specificities

    def classification_accuracy(self):
        cm = confusion_matrix(self.y_train, self.y_pred)
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

        labels = np.unique(self.y_train)
        print("\nClassification accuracy (CA):")
        for label, ca in zip(labels, per_class_CA):
            print(f"Class {label}: {ca:.3f}")

    def one_vs_rest_ROC_AUC(self, target, plot : bool = True):
        ''' Compute AUC for a specific target class using model probability scores (y_probs) '''
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(self.y_train)

        # Ensures that your model has already computed predict_proba() scores
        if hasattr(self, "y_probs"):
            y_probs = self.y_probs
        else:
            raise ValueError("Missing model scores (y_probs), evaluate first!")

        ''' Binary vs multiclass classification case '''
        if len(lb.classes_) == 2:
            if target not in lb.classes_:
                raise ValueError(f"Target '{target}' not in training labels.")

            target_index = np.flatnonzero(lb.classes_ == target)[0]
            y_true = (self.y_train == target)

            # Choose correct score column
            if y_probs.ndim == 1:
                y_probs_target = y_probs  # Already binary score
            elif y_probs.shape[1] == 2:
                y_probs_target = y_probs[:, target_index]
            else:
                raise ValueError("Unexpected y_probs shape for binary classification.")

        # Multiclass case
        else:
            if target not in lb.classes_:
                raise ValueError(f"Target '{target}' not in training labels.")

            class_id = np.flatnonzero(lb.classes_ == target)[0]
            if class_id >= y_probs.shape[1]:
                raise ValueError(f"Class ID {class_id} out of range in model scores.")

            y_true = y_bin[:, class_id]
            y_probs_target = y_probs[:, class_id]

        # Compute AUC
        auc = roc_auc_score(y_true, y_probs_target)

        # Plot ROC Curve
        if plot:
            RocCurveDisplay.from_predictions(
                y_true,  # True labels (one-hot)
                y_probs_target,  # Model scores (probabilities or decision function)
                name=f"{target} vs rest",
                color="darkorange",
                plot_chance_level=True,
                despine=True,
            )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve for {target} using {self.classifier_name}")
            plt.grid(True)
            plt.show()

        return auc

    def plot_precision_recall(self):
        """
        Nariše Precision-Recall krivuljo za en model.
        y_test   ... prave oznake (0/1 ali multi-label binarized)
        y_score  ... verjetnosti ali decision scores
        """

        precision, recall, _ = precision_recall_curve(self.y_train, self.y_probs)
        avg_prec = average_precision_score(self.y_train, self.y_probs)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"{self.classifier_name} (AP = {avg_prec:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve – {self.classifier_name}")
        plt.legend(loc="best")
        plt.grid(alpha=0.6)
        plt.show()

    def one_vs_rest_ROC_AUC_all_targets(self, plot : bool = True):
        ''' Calculates macro and weighted ROC curve '''
        # Ensures that your model has already computed predict_proba() scores
        if not hasattr(self, "y_probs"):
            raise ValueError("Missing model scores (y_probs), evaluate first!")

        lb = LabelBinarizer()
        y_onehot_true = lb.fit_transform(self.y_train)
        class_labels = lb.classes_  # Get class names

        # Binary classification case
        if len(class_labels) == 2:
            # y_probs can be shape (n_samples,) or (n_samples, 2)
            if self.y_probs.ndim == 1:
                y_probs_class1 = self.y_probs
            else:
                y_probs_class1 = self.y_probs[:, 1]

            auc_val = roc_auc_score(y_onehot_true, self.y_probs[:, 1])

            if plot:
                fpr, tpr, _ = roc_curve(y_onehot_true, y_probs_class1)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, color="blue", label=f"AUC = {auc_val:.3f}")
                plt.plot([0, 1], [0, 1], "k--", lw=1)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve ({class_labels[1]} vs {class_labels[0]})")
                plt.legend()
                plt.grid(True)
                plt.show()

            return {
                "AUC macro": auc_val # AUC binary
            }

        # Multiclass
        else:
            # Initialize variables for weighted ROC
            weighted_fpr = np.linspace(0, 1, 100)  # Interpolation points
            weighted_tpr = np.zeros_like(weighted_fpr)
            class_counts = np.sum(y_onehot_true, axis=0)  # Class distribution

            if plot:
                plt.figure(figsize=(7, 5))

            # Compute ROC for each class (OvR approach)

            for i, class_label in enumerate(class_labels):
                fpr, tpr, _ = roc_curve(y_onehot_true[:, i], self.y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                class_weight = class_counts[i] / len(self.y_train)  # Weight by class prevalence

                # Interpolate TPR at common FPR points
                interp_tpr = np.interp(weighted_fpr, fpr, tpr)
                weighted_tpr += interp_tpr * class_weight  # Weighted sum

                # Plot individual class ROC curves
                if plot:
                    plt.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.3f}) [{self.classifier_name}]")

            # Plot the final weighted ROC curve
            if plot:
                plt.plot(weighted_fpr, weighted_tpr, color='black', linestyle="--", label="Weighted ROC Curve")
                plt.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Weighted ROC Curve (One-vs-Rest)")
                plt.legend()
                plt.show()

        # Overall Weighted AUC
            macro_auc = roc_auc_score(y_onehot_true, self.y_probs, average="macro", multi_class="ovr")
            weighted_auc = roc_auc_score(y_onehot_true, self.y_probs, average="weighted", multi_class="ovr")
        #print(f"Macro-Averaged AUC: {macro_auc:.3f}")
        #print(f"Weighted-Averaged AUC: {weighted_auc:.3f}")

            return {
                "AUC macro": macro_auc,
                "AUC weighted": weighted_auc
            }



