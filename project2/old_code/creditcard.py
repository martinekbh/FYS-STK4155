import os
import numpy as np
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

from own_code import *

random.seed(0) # set seed


def import_data(filename):
    nanDict = {}

    # Import data into dataframe
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    for col in df.columns:      # Print variable names
        print(col)

    # Features and targets 
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    #print(X[0:2,:], '\n')

    n = len(y)          # Number of observations
    p = np.shape(X)[-1] # Number of explanatory variables

    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto")

    X = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough"
    ).fit_transform(X)

    #print(X[0:2,:])
    #y.shape

    # Train-test split
    trainingShare = 0.5 
    seed  = 1
    XTrain, XTest, yTrain, yTest=train_test_split(X, y, train_size=trainingShare, \
                                          test_size = 1-trainingShare,
                                                     random_state=seed)

    # Input Scaling
    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    # One-hot's of the target vector
    Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)

    # Remove instances with zeros only for past bill statements or paid amounts
    df = df.drop(df[(df.BILL_AMT1 == 0) &
            (df.BILL_AMT2 == 0) &
            (df.BILL_AMT3 == 0) &
            (df.BILL_AMT4 == 0) &
            (df.BILL_AMT5 == 0) &
            (df.BILL_AMT6 == 0)].index)

    df = df.drop(df[(df.PAY_AMT1 == 0) &
            (df.PAY_AMT2 == 0) &
            (df.PAY_AMT3 == 0) &
            (df.PAY_AMT4 == 0) &
            (df.PAY_AMT5 == 0) &
            (df.PAY_AMT6 == 0)].index)
    return


cwd = os.getcwd() # Current working directory
filename = cwd + '/default of credit card clients.xls'
import_data(filename)


# REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lambdas=np.logspace(-5,7,13)
parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
scoring = ['accuracy', 'roc_auc']
logReg = LogisticRegression()
gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc') 
print(gridSearch)


