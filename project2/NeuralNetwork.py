import numpy as np

class NeuralNetwork:
    def __init__(self, X, y, 
        n_layers = 1,
        n_hidden_neurons=[50], 
        n_categories = 2, 
        epochs=10, batch_size=100, 
        eta=0.1, lmbd=0.0, 
        problem='class',
        activation='sigmoid',
        seed = None):
        
        # data
        self.X = X
        self.y = y              # y must be the shape of OneHotEncoder
        self.X_full_data = X
        self.y_full_data = y

        # other vars
        self.n_inputs, self.n_features = X.shape
        self.n_layers = n_layers
        self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.problem = problem
        self.activation = activation

        # Set number of hidden neurons
        if n_hidden_neurons == None:
            self.n_hidden_neurons = [50]*n_layers
        elif isinstance(n_hidden_neurons, list) and len(n_hidden_neurons)==n_layers:
            self.n_hidden_neurons = n_hidden_neurons
        else:
            print(isinstance(n_hidden_neurons, list))
            print(len(n_hidden_neurons)==n_layers)
            print(n_layers)
            print(len(n_hidden_neurons))
            msg = "The arg n_hidden_neurons must be a list of integers," \
                    "and must have length equal to the arg n_layers."
            raise ValueError(msg)


        # Set seed
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
        self.output_bias = np.zeros((self.n_categories, 1)) + 0.01

    def feed_forward(self):
        """ Feed forward for training """
        self.z_h = []
        self.a_h = []
        a = self.X

        for l in range(self.n_layers):
            #print(f"layer {l}")
            #z = (self.hidden_weights[l].T @ a_h) + self.hidden_bias[l]
            #self.z_h.append(z)
            z = np.matmul(a, self.hidden_weights[l]) + self.hidden_bias[l].T
            a = self.activation_func(z)
            self.a_h.append(a)

        #self.z_o = (self.output_weights.T @ a_h) + self.output_bias.T
        self.z_o = np.matmul(a, self.output_weights) + self.output_bias.T
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / (np.sum(exp_term, axis=1, keepdims=True))

    def feed_forward_out(self, X):
        """ Feed forward for output """
        a_h = X

        for l in range(self.n_layers):
            z = np.matmul(a_h, self.hidden_weights[l]) + self.hidden_bias[l].T
            a_h = self.activation_func(z)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias.T
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities


    def activation_func(self, x):
        if self.activation == 'sigmoid':
            return 1/(1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)

    def activation_derivative(self, a):
        # Return derivative of activation function a = f(z)
        if self.activation == 'sigmoid':
            return a*(1-a)

    def back_propagation(self):
        a_L = self.probabilities
        error_output = a_L * (1 - a_L )*(a_L - self.y)
        a_h = self.a_h[-1]

        # The gradients of the outputs
        self.output_weights_gradient = np.matmul(a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)            # Is reshaped
        self.output_bias_gradient = self.output_bias_gradient.reshape(len(self.output_bias_gradient),1)

        # Make empty lists
        error_hidden = []
        self.hidden_weights_gradient = []
        self.hidden_bias_gradient = []
        
        # Calculate error and gradients of the hidden layers
        err = np.matmul(error_output, self.output_weights.T) * a_h * (1 - a_h)
        for l in range((self.n_layers-2), -1, -1):
            #print(f"l: {l}")
            error_hidden.insert(0,err) 
            self.hidden_weights_gradient.insert( 0, np.matmul(self.a_h[l].T, err) )
            self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]), 1) )

            err = np.matmul(err, self.hidden_weights[l+1].T) * self.a_h[l] * (1 - self.a_h[l]) 

        self.hidden_weights_gradient.insert( 0, np.matmul(self.X.T, err) )
        self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]),1) )

        # Regression parameter
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            for i in range(len(self.hidden_weights_gradient)): # Maybe -1 on the range?
                self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]

        # Update the weights and biases 
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient[i]
            self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient[i]

    def back_propagation_classification(self):
        a_L = self.probabilities
        error_output = a_L - self.y     # delta_L
        a_h = self.a_h[-1]

        # The gradients of the outputs
        self.output_weights_gradient = np.matmul(a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)            # Is reshaped
        self.output_bias_gradient = self.output_bias_gradient.reshape(len(self.output_bias_gradient),1)

        # Make empty lists
        error_hidden = []
        self.hidden_weights_gradient = []
        self.hidden_bias_gradient = []
        
        # Calculate error and gradients of the hidden layers
        f_z_derived = self.activation_derivative(a_h)
        err = np.matmul(error_output, self.output_weights.T) * f_z_derived
        for l in range((self.n_layers-2), -1, -1):
            error_hidden.insert(0,err) 
            self.hidden_weights_gradient.insert( 0, np.matmul(self.a_h[l].T, err) )
            self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]), 1) )
            f_z_derived = self.activation_derivative(self.a_h[l])
            err = np.matmul(err, self.hidden_weights[l+1].T) * f_z_derived

        self.hidden_weights_gradient.insert( 0, np.matmul(self.X.T, err) )
        self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]),1) )

        # Regression parameter
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            for i in range(len(self.hidden_weights_gradient)): # Maybe -1 on the range?
                self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]

        # Update the weights and biases 
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient[i]
            self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient[i]

    def back_propagation_regression(self):
        a_L = self.probabilities
        error_output = a_L*(1- a_L)*(a_L - self.y)     # delta_L
        a_h = self.a_h[-1]

        # The gradients of the outputs
        self.output_weights_gradient = np.matmul(a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)            # Is reshaped
        self.output_bias_gradient = self.output_bias_gradient.reshape(len(self.output_bias_gradient),1)

        # Make empty lists
        error_hidden = []
        self.hidden_weights_gradient = []
        self.hidden_bias_gradient = []
        
        # Calculate error and gradients of the hidden layers
        f_z_derived = self.activation_derivative(a_h)
        err = np.matmul(error_output, self.output_weights.T) * f_z_derived
        for l in range((self.n_layers-2), -1, -1):
            error_hidden.insert(0,err) 
            self.hidden_weights_gradient.insert( 0, np.matmul(self.a_h[l].T, err) )
            self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]), 1) )
            f_z_derived = self.activation_derivative(self.a_h[l])
            err = np.matmul(err, self.hidden_weights[l+1].T) * f_z_derived

        self.hidden_weights_gradient.insert( 0, np.matmul(self.X.T, err) )
        self.hidden_bias_gradient.insert( 0, np.sum(err, axis=0).reshape(len(err[0]),1) )

        # Regression parameter
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            for i in range(len(self.hidden_weights_gradient)): # Maybe -1 on the range?
                self.hidden_weights_gradient[i] += self.lmbd * self.hidden_weights[i]

        # Update the weights and biases 
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient[i]
            self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient[i]

    def sgd(self, n_epochs, n_minibatches=None):
        Xtrain = self.Xtrain; Xtest = self.Xtest; ytrain = self.ytrain; ytest = self.ytest
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
                gradient = -xi.T @ (yi - p) 
                l = self.learning_schedule(epoch*n_minibatches + i)
                beta = beta - l * gradient
                self.beta = beta

        return beta 

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)
        
        if self.problem == "class":
            for i in range(self.epochs):
                for j in range(self.iterations):
                    # Pick datapoint with repacement:
                    chosen_datapoints = np.random.choice(
                            data_indices, size=self.batch_size, replace=False)
                    # minibatch training data
                    self.X = self.X_full_data[chosen_datapoints]
                    self.y = self.y_full_data[chosen_datapoints]
                    self.feed_forward()
                    self.back_propagation_classification()

        if self.problem == "reg":
            for i in range(self.epochs):
                for j in range(self.iterations):
                    # Pick datapoint with repacement:
                    chosen_datapoints = np.random.choice(
                            data_indices, size=self.batch_size, replace=False)
                    # minibatch training data
                    self.X = self.X_full_data[chosen_datapoints]
                    self.y = self.y_full_data[chosen_datapoints]
                    self.feed_forward()
                    self.back_propagation_regression()

