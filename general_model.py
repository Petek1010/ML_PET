from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict


class GeneralModel:
    def __init__(self, X_train, y_train, X_test, y_test, model, param_grid=None, scoring="accuracy"):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.best_model = None
        self.best_params = None

    def search_best_params(self):
        if self.param_grid:
            grid_search = GridSearchCV(self.model, self.param_grid, cv=LeaveOneOut(), scoring=self.scoring, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best Params for {self.model.__class__.__name__}: {self.best_params}")
            print('best estimator', self.best_model)
        else:
            self.best_model = self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        if self.best_model is None:
            raise ValueError("Model is not trained. Run tune_hyperparameters() first.")

        self.y_pred = cross_val_predict(self.best_model, self.X_train, self.y_train, cv=LeaveOneOut())
        print('\n Classification report for',self.model.__class__.__name__ ,': \n', classification_report(self.y_train, self.y_pred, digits=3))

        score = self.best_model.score(self.X_train, self.y_train) #before: X_test, y_test
        print(f"{self.model.__class__.__name__} Test Accuracy: {score:.3f}")

        return score

    def predict(self):
        """Predict on test data."""
        if self.best_model is None:
            raise ValueError("Model is not trained. Run tune_hyperparameters() first.")
        return self.best_model.predict(self.X_test)