from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from general_model import GeneralModel


class SVM(GeneralModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        param_grid = {
            "C": [0.9, 1, 10],
            "kernel": ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': [1, 2, 3, 4],
            'gamma': ['scale', 'auto']
        }
        super().__init__(X_train, y_train, X_test, y_test, SVC(probability=True, random_state=42), param_grid)



class NaiveBayes(GeneralModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test, CategoricalNB(), None)