import sys
import svm


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets



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

    print(train_data)

    return train_data, test_data




if __name__ == '__main__':
    print("Start program")

    rawData = importData("AllScores.xlsx", "ZScores")
    train_data, test_data = filterData(rawData)
    #svm.SVM(train_data,test_data)
    svm.cross_val_svm(train_data, test_data)

    ''' Visualizing '''



