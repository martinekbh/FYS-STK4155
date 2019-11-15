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

print(X.shape)


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

hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01
print(f"hidden weights: {hidden_weights.shape}, hidden bias: {len(hidden_bias)}")

# weight and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01
print(f"output weights: {output_weights.shape}, output bias: {output_bias.shape}")

def feed_forward(X):
    """
    Feed-Forward Neural Network with 1 hidden layer
    (Single Layer Perception, i.e. SLP)
    """
    # weight and bias in the hidden layer:


    z_h = np.matmul(X, hidden_weights) + hidden_bias # zh = XW + b
    a_h = sigmoid(z_h)              # activation in the hidden layer
    print(f"z_h: {z_h.shape}")

    z_o = np.matmul(a_h, output_weights) + output_bias

    # softmax product
    probabilities = np.exp(z_o)/(np.sum( np.exp(z_o), axis=1, keepdims=True ))
    return a_h, probabilities

def feed_forward2(X, n_layers, n_hidden_neurons):
    """ 
    Feed forward for training 
    MLP 
    """

    """
    W = [np.random.randn(n_features, n_hidden_neurons[0])]
    b = [np.zeros((n_hidden_neurons[0], 1)) + 0.01]
    for l in range(1, n_layers):
        W.append( np.random.randn(n_hidden_neurons[l-1], n_hidden_neurons[l]) )
        b.append( np.zeros((n_hidden_neurons[l], 1)) + 0.01 )

    #self.hidden_bias = np.zeros((self.n_layers, self.n_hidden_neurons, 1)) + 0.01
    hidden_weights = W
    hidden_bias = b
    print(f"hidden weights: {len(hidden_weights)}, {hidden_weights[0].shape}, hidden bias: {len(hidden_bias[0])}")

    output_weights = np.random.randn(n_hidden_neurons[-1], n_categories)
    output_bias = np.zeros((n_categories,1)) + 0.01
    print(f"output weights: {output_weights.shape}, output bias: {output_bias.shape}")
    """

    z_h = []
    a_h = X

    for l in range(n_layers):
        print(f"layer {l}")
        #print(f"W: {hidden_weights[l].shape}")
        #print(f"a: {a_h.shape}")
        #print(f"b: {hidden_bias[l].shape}")
        #z = (hidden_weights[l].T @ a_h) + hidden_bias[l]
        z = np.matmul(a_h, hidden_weights[l]) + hidden_bias[l].T
        print(f"z_h: {z.shape}")
        z_h.append(z)
        a_h = sigmoid(z)

    #z_o = (output_weights.T @ a_h) + output_bias
    z_o = np.matmul(a_h, output_weights) + output_bias.T
    exp_term = np.exp(z_o)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    return probabilities

def predict(X, i=1):
    if i==1:
        probs = feed_forward(X)[1][:,1]    # Probabiliity of cancer

    else:
        probs = feed_forward2(X, n_layers = 1, n_hidden_neurons=[50])[:,1]
    # return index of class with highest probability
    #return np.argmax(probs, axis=1)
    return np.round(probs)



def backpropagation(X, Y):
    a_h, probabilities = feed_forward(X)
    
    # error in the output layer
    error_output = probabilities - Y
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient
    

def back_propagation2(y, probabilities):
    error_output = probabilities - y
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)

    self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
    self.output_bias_gradient = np.sum(error_output, axis=0)

    self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
    self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

    if self.lmbd > 0.0:
        self.output_weights_gradient += self.lmbd * self.output_weights
        self.hidden_weights_gradient += self.lmbd * self.hidden_weights

    self.output_weights -= self.eta * self.output_weights_gradient
    self.output_bias -= self.eta * self.output_bias_gradient
    self.hidden_weights -= self.eta * self.hidden_weights_gradient
    self.hidden_bias -= self.eta * self.hidden_bias_gradient


print("\nFeed-Forward method (SLP)")
pred = predict(X_test)
#print(pred[:10])
#print(y_test[:10])
acc = np.sum((pred == y_test))/len(y_test)
print(acc)
dWo, dBo, dWh, dBh = backpropagation(X_test, y_test)

hidden_weights = [hidden_weights]
hidden_bias = [hidden_bias]
output_bias = output_bias.reshape((n_categories,1))


print("\n\nFeed-Forward method 2")
pred = predict(X_test, i=2)
acc = np.sum((pred == y_test))/len(y_test)
print(acc, "\n\n")


class NeuralNetwork2:
    def __init__(self, X, y, 
            n_layers = 2,
            n_hidden_neurons=(50,50), 
            n_categories=2, 
            epochs=10, batch_size=100, 
            eta=0.1, lmbd=0.0, 
            seed = None):
        
        self.X = X
        self.y = y
        self.n_inputs, self.n_features = X.shape
        self.n_layers = n_layers
        self.n_categories = n_categories
        self.n_hidden_neurons = n_hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd

        if seed != None:
            np.random.seed(seed)

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        W = [np.random.randn(self.n_features, self.n_hidden_neurons[0])]
        b = [np.zeros((self.n_hidden_neurons[0], 1)) + 0.01]
        for l in range(1, self.n_layers):
            W.append( np.random.randn(self.n_hidden_neurons[l-1], self.n_hidden_neurons[l]) )
            b.append( np.zeros((self.n_hidden_neurons[l], 1)) + 0.01 )

        #self.hidden_bias = np.zeros((self.n_layers, self.n_hidden_neurons, 1)) + 0.01
        self.hidden_weights = W
        self.hidden_bias = b

        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)
        self.output_bias = np.zeros((self.n_categories,1)) + 0.01

    def feed_forward(self):
        """ Feed forward for training """
        self.z_h = []
        a_h = self.X.T

        for l in range(self.n_layers):
            print(f"layer {l}")
            print(f"W: {self.hidden_weights[l].shape}")
            print(f"a: {a_h.shape}")
            print(f"b: {self.hidden_bias[l].shape}")
            z = (self.hidden_weights[l].T @ a_h) + self.hidden_bias[l]
            self.z_h.append(z)
            #self.z_h.append(np.matmul(a_h, self.hidden_weights[l]) + self.hidden_bias[l])
            a_h = self.sigmoid(z)

        #self.z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        self.z_o = (self.output_weights.T @ a_h) + self.output_bias
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        """ Feed forward for output """
        #z_h = []
        a_h = X.T

        for l in range(self.n_layers):
            print(f"layer {l}")
            print(f"W: {self.hidden_weights[l].shape}")
            print(f"a: {a_h.shape}")
            print(f"b: {self.hidden_bias[l].shape}")
            z = (self.hidden_weights[l].T @ a_h) + self.hidden_bias[l]
            #z_h.append(z)
            #self.z_h.append(np.matmul(a_h, self.hidden_weights[l]) + self.hidden_bias[l])
            a_h = self.sigmoid(z)

        #self.z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        z_o = (self.output_weights.T @ a_h) + self.output_bias
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities


    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def back_propagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient


    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=0)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities



class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=2,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]             # Number of observations
        self.n_features = X_data.shape[1]           # Number of predictors
        self.n_hidden_neurons = n_hidden_neurons    # Number of neurons in hidden layer
        self.n_categories = n_categories            # Number of groups

        self.epochs = epochs            # Number of epochs in SGD
        self.batch_size = batch_size    # Batch size in sgd
        self.iterations = self.n_inputs // self.batch_size  # Number of iterations
        self.eta = eta                  # Learning rate
        self.lmbd = lmbd                # Regularization parameter lambda

        self.create_biases_and_weights()    # Make weights and biases

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)     # indexes of data-point

        for i in range(self.epochs):                # Loop over epochs
            for j in range(self.iterations):        # Loop over iterations
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


y_train_vec = y_train.reshape(len(y_train), 1)

ffnn = NeuralNetwork(X_train, y_train_vec)
ffnn.train()
pred = ffnn.predict(X_test)
acc = np.sum((pred == y_test))
print("Own Neural Network class (SLP)")
print(acc)


# SCIKIT LEARN
from sklearn.neural_network import MLPClassifier
lmbd = 0
eta = 0.1
epochs = 100
dnn = MLPClassifier(hidden_layer_sizes = n_hidden_neurons, activation='logistic',
                    alpha = lmbd, learning_rate_init=eta, max_iter=epochs)

#print(f"Scikit number of layers: {dnn.n_layers}")
dnn.fit(X_train, y_train)
print("SKIKIT learn MLPClassifier:")
print(dnn.score(X_test, y_test))
