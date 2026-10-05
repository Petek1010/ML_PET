from unittest.mock import inplace

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_excel("AllScores.xlsx", "ZScores")


def preprocess_data(df):
    df.drop(columns=["SubjectName", "ADRP_NY_gis002","CBDRP1010_NY_gis001","MSA_NY_gis001",	"PDRP_NY_gis001",
                     "PSPRP_NY_gis001",		"bFDRP_NY_gis001"
    ], inplace=True)

    df = df.loc[df['FirstThreeChars'].isin(["AD_", "CBD", "CJD", "DLB", "FTD", "MSA", "NC-", "PD-", "PSP"])]

    return df

data = preprocess_data(data)

X = data.drop(column=["FirsThreeChars"])
y = data["FirsThreeChars"]



scaler = StandardScaler()
