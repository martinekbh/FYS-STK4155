#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
from own_code import *

mnist = tf.keras.datasets.mnist     # Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale data (min-max scaling)
x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test))/(np.max(x_test) - np.min(x_test))

# Print info about dataset
print("\nINFO:")
print("The MNIST dataset with images of handwritten digits")
print(f"Total number of observations: {len(y_train) + len(y_test)}")
print(f" - Of which {len(y_train)} are in the training set,")
print(f"   and {len(y_test)} are in the test set")
print(f"The shape of each image (X-value) is {x_train[0].shape} pixels\n")

# Visualize a number in the dataset
image_index = 111
plt.imshow(x_train[image_index], cmap='Greys')
plt.title(f"Example of handwritten {y_train[image_index]}")
save_fig("example_image", folder="mnist", extension="pdf")
save_fig("example_image", folder="mnist", extension="png")
plt.show()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28,28,1)


# Build MLP
model = tf.keras.Sequential()
# Add dense connected layer with 64 units:
model.add(layers.Dense(60, activation='relu'))
# Add one more
model.add(layers.Dense(60, activation='relu'))
# Add output layer
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(0.01),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = [tf.keras.metrics.CategoricalAccuracy()])


print("Fit model on training data")
fit = model.fit(x_train, y_train,
                batch_size=120,
                epochs=3)

print("fit: ", fit.history) # The history holds record of loss and metric values during training
print(model.summary())
print("\nEvaluate on test data: ")
results = model.evaluate(x_test, y_test, batch_size=120)
print("test loss, test acc: ", results)
