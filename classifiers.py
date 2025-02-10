from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Base Class for Models
class BaseModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        """Train the model with standardized data."""
        self.pipeline = make_pipeline(StandardScaler(), self.model)
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using the trained model."""
        return self.pipeline.predict(X_test)

    def get_params(self):
        """Returns the model parameters."""
        return self.model.get_params()

class SVM(BaseModel):
    def __init__(self, kernel='linear', C=1.0, probability=True, decision_function_shape='ovr'):
        super().__init__()
        self.kernels = {'linear', 'poly', 'rbf', 'sigmoid'}
        # Validate kernel choice
        if kernel in self.kernels:
            self.kernel = kernel
        else:
            raise ValueError(f"Invalid kernel '{kernel}'. Choose from {self.kernels}")
        self.model = SVC(kernel=self.kernel,
                         C=C,
                         probability=probability,
                         decision_function_shape=decision_function_shape)

