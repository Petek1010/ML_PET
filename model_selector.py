from models import SVM, NaiveBayes


class ModelSelector:
    def __init__(self, X_train, y_train, X_test, y_test, scoring="accuracy"):
        self.models = [
            SVM(X_train, y_train, X_test, y_test),
            NaiveBayes(X_train, y_train, X_test, y_test)
        ]

        self.scoring = scoring
        self.best_model = None
        self.best_score = -float("inf")

    def find_best_model(self):
        """Train and evaluate each model, then select the best one."""
        for model in self.models:
            model.search_best_params()
            score = model.evaluate()
            if score > self.best_score:
                self.best_score = score
                self.best_model = model

        print(f"\nBest Model: {self.best_model.model.__class__.__name__} with score: {self.best_score:.3f}")

    def predict(self):
        """Predict using the best model."""
        if self.best_model is None:
            raise ValueError("No model has been selected. Run find_best_model() first.")
        return self.best_model.predict()