# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def accuracy(actual, pred):
    return np.sum( actual == pred )/len(actual)

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))
 
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat) # intercept
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)
  
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
  
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Test the logistic regression algorithm on the diabetes dataset
seed(1)

# load and prepare cancer data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
dataset = np.zeros((len(X), len(X[0])+1))
dataset[:,:-1] = X; dataset[:,-1] = y
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
train, test, trash, trash = train_test_split(dataset, y, random_state=0)
y_test = test[:,-1]
X_test = test[:,:-1]
X_train = train[:,:-1]
y_train = train[:,-1]

"""
# Set up training data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
scaler = StandardScaler() # Scale data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
temp = np.zeros((len(X_train), len(X_train[0])+1))
temp[:,:-1] = X_train; temp[:,-1] = y_train
train = temp
temp = np.zeros((len(X_test), len(X_test[0])+1))
temp[:,:-1] = X_test; temp[:,-1] = y_test
test = temp
"""

# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
#pred = logistic_regression(train, test, l_rate, n_epoch)
#acc = accuracy_metric(y_test, pred)
#print("accuracy score:", acc)


#scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)

def mSGD(Xtrain, ytrain, n_epochs, n_minibatches=None, seed=1):
    np.random.seed(seed)
    n = len(Xtrain)

    if n_minibatches == None: # Default number of minibatches is n
        n_minibatches = n
        batch_size = 1

    else:
        batch_size = int(n / n_minibatches)

    beta = np.random.randn(len(Xtrain[0]), 1) # why random?

    for epoch in range(n_epochs):   # epoch

        j = 1
        for i in range(n_minibatches):          # minibatches
            random_index = np.random.randint(n_minibatches)

            xi = Xtrain[random_index * batch_size: random_index*batch_size + batch_size]
            yi = ytrain[random_index * batch_size: random_index*batch_size + batch_size]
            yi = yi.reshape((batch_size, 1))

            p = 1/(1 + np.exp(-xi @ beta))
            
            if p.shape != yi.shape:         # Test if something is wrong
                print("\nOOPS")
                print("xi:", xi.shape)
                print("beta:", beta.shape)
                print("index:", random_index)
                print("x*b:", (xi@beta).shape)
                print("p:", p.shape)
                print("yi:", yi.shape)
                print(p)
                exit()

            gradient = -xi.T @ (yi - p) 
            l = learning_schedule(epoch*n_minibatches + i)
            beta = beta - l * gradient

    return beta

intercep = np.ones((len(X[:,0]),1))
X_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((len(X_test), 1)), X_test), axis=1)
beta = mSGD(X_train, y_train, n_epoch, n_minibatches = 30, seed=1001)
pred = np.round(1/(1 + np.exp(-X_test@beta))).ravel()
acc = np.sum( y_test == pred )/len(y_test)
print(acc)


