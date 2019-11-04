import numpy as np
from random import seed
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import StandardScaler

np.random.seed(1)   # Set seed

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

n_inputs = len(y_train)
n_features = len(X_train[0])
n_hidden_neurons = 50
n_categories = 2

# weight and bias in the hidden layer:
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weight and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

def feed_forward(X):
    """
    Feed-Forward Neural Network with 1 hidden layer
    (Single Layer Perception, i.e. SLP)
    """
    z_h = np.matmul(X, hidden_weights) + hidden_bias # zh = XW + b
    a_h = sigmoid(z_h)              # activation in the hidden layer

    z_o = np.matmul(a_h, output_weights) + output_bias

    # softmax product
    probabilities = np.exp(z_o)/(np.sum( np.exp(z_o), axis=1, keepdims=True ))
    return probabilities



def predict(X):
    probs = feed_forward(X)[:,1]    # Probabiliity of cancer
    # return index of class with highest probability
    #return np.argmax(probs, axis=1)
    return np.round(probs)


pred = predict(X_test)
print(pred[:10])
print(y_test[:10])
acc = np.sum((pred == y_test))/len(y_test)
print(acc)
