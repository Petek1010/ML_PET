from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    recall_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score


class Classifier:
    """
    Chose desired classifier and its hyperparameters. If no parameters are set, defaults will be used.

    Default parameters:

    - SVM: kernel='linear', C=0.9, decision_function_shape="ovr", random_state=42, probability=True

    - Logistic Regression: penalty='l2', C=4.0, solver='lbfgs', random_state=42
    """
    def __init__(self, X_train, y_train, X_test, y_test, classifier_name="svm", **kwargs):
        self.classifiers = {
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "naive_bayes": CategoricalNB,
            "random_forest": RandomForestClassifier,
            "knn": KNeighborsClassifier
        }

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classifier_name = classifier_name

        self.classifier = self.classifiers[classifier_name](**kwargs) if kwargs else self._set_default_params()

    def _set_default_params(self):
        default_params = {
            "svm": {"kernel": "linear", "C": 0.9, "decision_function_shape": "ovr", "random_state": 42, "probability": True},
            "logistic_regression": {"penalty": "l2", "C": 4.0, "solver": "lbfgs", "random_state": 42},
            "naive_bayes": {},
        }
        return self.classifiers[self.classifier_name](**default_params.get(self.classifier_name,{}))

    def set_best_parameters(self):
        best_params = self.grid_parameter_search()
        if self.classifier_name == 'svm':
            self.classifier = SVC(**best_params, decision_function_shape="ovr", random_state=42, probability=True)
        elif self.classifier_name == 'logistic_regression':
            self.classifier = LogisticRegression(**best_params, solver='lbfgs', random_state=42)

    def get_params(self):
        return self.classifier.get_params()

    def evaluate(self, cv=LeaveOneOut(), result='prediction'):
        self.y_pred = cross_val_predict(self.classifier, self.X_train, self.y_train, cv=cv)
        self.y_score = cross_val_predict(self.classifier, self.X_train, self.y_train, cv=cv, method="predict_proba")

        if self.classifier_name == 'naive_bayes':
            print('Not Yet implemented')
            self.y_pred = cross_val_predict(self.classifier, self.X_train_discrete, self.y_train, cv=cv)

        if result == 'prediction':
            print('\n Classification report for [', self.classifier_name,']: \n', classification_report(self.y_train, self.y_pred, digits=3))
        elif result == 'probability':
            print('probability for first row: ', self.y_score[0])

    def get_model_evaluation(self):
        # ORANGE params
        self.classification_accuracy()
        recall = recall_score(self.y_train, self.y_pred, average='macro')  # or 'micro', 'weighted'
        print(f'Overall Recall (average): {recall:.3f}')



        # Compute multiclass AUC
        #auc = roc_auc_score(self.y_train, self.y_score, multi_class='ovr', average='weighted')
        #print(f'Multiclass AUC (OvR) average over all labels: {auc:.3f}')


    def predict(self):
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier.predict(self.X_test)
       

    def grid_parameter_search(self, cv=LeaveOneOut()):
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


    def confusion_matrix(self):
        cm = confusion_matrix(self.y_train, self.y_pred)
        labels = np.unique(self.y_train)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Greens")
        ax.set_title("Confusion Matrix")
        plt.show()

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

    def ROC_analysis_target(self, target):
        lb = LabelBinarizer()
        y_onehot_true = lb.fit_transform(self.y_train)

        # Select class of interest for one-vs-rest ROC
        class_of_interest = target
        class_id = np.flatnonzero(lb.classes_ == class_of_interest)[0]
        print(f"Class ID for {class_of_interest}: {class_id}")

        # Plot ROC Curve
        display = RocCurveDisplay.from_predictions(
            y_onehot_true[:, class_id],  # True labels (one-hot)
            self.y_score[:, class_id],  # Model scores (probabilities or decision function)
            name=f"{class_of_interest} vs the rest",
            color="darkorange",
            plot_chance_level=True,
            despine=True,
        )

        # Improve Plot Appearance
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"One-vs-Rest ROC Curve:\n{class_of_interest} vs All",
        )
        plt.show()

    def ROC_analysis_all_targets(self):
        # Convert y_train to one-hot encoding
        lb = LabelBinarizer()
        y_onehot_true = lb.fit_transform(self.y_train)
        class_labels = lb.classes_  # Get class names

        # Initialize variables for weighted ROC
        weighted_fpr = np.linspace(0, 1, 100)  # Interpolation points
        weighted_tpr = np.zeros_like(weighted_fpr)
        class_counts = np.sum(y_onehot_true, axis=0)  # Class distribution

        # Compute ROC for each class (OvR approach)
        plt.figure(figsize=(7, 5))
        for i, class_label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_onehot_true[:, i], self.y_score[:, i])
            roc_auc = auc(fpr, tpr)
            class_weight = class_counts[i] / len(self.y_train)  # Weight by class prevalence

            # Interpolate TPR at common FPR points
            interp_tpr = np.interp(weighted_fpr, fpr, tpr)
            weighted_tpr += interp_tpr * class_weight  # Weighted sum

            # Plot individual class ROC curves
            plt.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.3f})")

        # Plot the final weighted ROC curve
        plt.plot(weighted_fpr, weighted_tpr, color='black', linestyle="--", label="Weighted ROC Curve")
        plt.plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Weighted ROC Curve (One-vs-Rest)")
        plt.legend()
        plt.show()

        # Overall Weighted AUC
        macro_auc = roc_auc_score(y_onehot_true, self.y_score, average="macro", multi_class="ovr")
        weighted_auc = roc_auc_score(y_onehot_true, self.y_score, average="weighted", multi_class="ovr")
        print(f"Macro-Averaged AUC: {macro_auc:.3f}")
        print(f"Weighted-Averaged AUC: {weighted_auc:.3f}")



