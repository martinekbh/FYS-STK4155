import os
import numpy as np
import random
import pandas as pd

random.seed(0) # set seed

cwd = os.getcwd() # Current working directory
filename = cwd + '/default of credit card clients.xls'
nanDict = {}

# Import data into dataframe
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Features and targets 
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


