from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import classifiers
from classifiers import SVM


class MultiClassifier:
    def __init__(self, model_name="svm", **kwargs):
        self.classifiers = {
            "logistic_regression": LogisticRegression(solver='saga', max_iter=1000, random_state=42),
            "svm": SVM,
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=5)
        }
        self.classifier = None
        self.set_classifier(model_name, **kwargs)


    def set_classifier(self, model_name, **kwargs):
        """Switches to a new model dynamically and applies parameters."""
        if model_name in self.classifiers:
            self.classifier = self.classifiers[model_name](**kwargs)
        else:
            raise ValueError(f"Invalid model name '{model_name}'. Choose from {list(self.classifiers.keys())}")

    def get_params(self):
        return self.classifier.get_params()

    def process_data(self, train_data):
        self.X_train = train_data.iloc[:, 1:].values  # features: vFDRP, ADRP, CJDRP
        self.y_train = train_data["FirstThreeChars"].values  # target: First three chars: AD, AD... (multiclass)

        # Normalize and scale train and test features
        scaler = StandardScaler()
        self.X_train_norm = scaler.fit_transform(self.X_train)


    def validate(self, cv=LeaveOneOut(), result='prediction'):
        if result == 'prediction':
            y_pred = cross_val_predict(self.model, self.X_train_norm, self.y_train, cv=cv)
            print('\n Classification report: \n',classification_report(self.y_train, y_pred, digits=3))
        elif result == 'probability':
            y_score = cross_val_predict(self.model, self.X_train_norm, self.y_train, cv=cv, method="predict_proba")






    def train(self, X_train, y_train):
        """Fits the model to the training data."""
        self.pipeline = make_pipeline(StandardScaler(), self.model)
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions using the trained model."""
        return self.pipeline.predict(X_test)

    def evaluate(self, X_train, y_train, cv=5):
        """Evaluates the model using cross-validation."""
        scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"{self.model_name} Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        return scores

    def set_model(self, model_name):
        """Switches the model dynamically."""
        if model_name in self.models:
            self.model_name = model_name
            self.model = self.models[model_name]
        else:
            print(f"Model '{model_name}' not found. Using Logistic Regression as default.")



