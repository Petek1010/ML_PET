import sys

import pandas as pd

from data_process import DataProcess, SVMPreprocess
from model_selector import ModelSelector
from models import SVM
from multiclassifier import MultiClassifier
from classifier import Classifier


def importData(fileName, sheetName = ''):
    '''Import all patients'''
    # TO DO: import data can read csv or excel file
    readData = pd.read_excel(fileName, sheet_name=sheetName)

    #print(readData.head())
    return readData


def filterData(data):

    firstThreeChars = data["FirstThreeChars"]
    ZScores = data.iloc[:,2:]
    features = ZScores.loc[:, ['bFDRP_SLO_gis001', 'ADRP_SLO__gis001', 'CJDRP_gis001_004']]

    filtered_data = pd.DataFrame()
    filtered_data["FirstThreeChars"] = firstThreeChars  # Add the firstThreeChars column
    filtered_data = pd.concat([filtered_data, features], axis=1)  # Add the selected features

    '''Choose patients for training and testing'''

    train_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['AD_', 'CJD', 'FTD', 'NC-'])]
    test_data = filtered_data.loc[filtered_data.FirstThreeChars.isin(['swd'])]




    return train_data, test_data




if __name__ == '__main__':
    print("Start program")
    #data = DataProcess("AllScores.xlsx", "ZScores")
    #X_train, y_train, X_test, y_test = data.old_pipeline()

    # Creating data for SVM
    svm_data = SVMPreprocess("AllScores.xlsx", "ZScores")
    X_train, y_train, X_test, y_test = svm_data.preprocess()

   # Testing model selector
   # model_selector = ModelSelector(X_train, y_train, X_test, y_test, scoring="accuracy")
   # model_selector.find_best_model()



    """ Supported vector machines """
    SVM_model = Classifier(X_train, y_train, X_test, y_test, 'svm')
    #SVM_model.evaluate()
    #SVM_model.get_model_evaluation()

    """ Logistic regression """
    lr_model = Classifier(X_train, y_train, X_test, y_test,'logistic_regression', C=4)
    lr_model.evaluate()
    lr_model.get_model_evaluation()

    """ Naive Bayes """


    #--------------------------------------------------------------------




    print(" OLDER VERSION AS A COMPARISON BASE\n")

    rawData = importData("AllScores.xlsx", "ZScores")
    train_data, test_data = filterData(rawData)

    svm_model = MultiClassifier(train_data, test_data, classifier_name='svm')
    svm_model.evaluate()
    #svm_model.ROC_analysis_all_targets()
    #lr_model = MultiClassifier(train_data, test_data, classifier_name='logistic_regression')

    #nb_model = MultiClassifier(train_data,test_data,classifier_name='naive_bayes')
    #nb_model.evaluate()
    #print(nb_model.get_params())
    #nb_model.confusion_matrix()











