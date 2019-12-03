#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
#from tf.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np
from own_code import *
import time

folder = "mnist"    # The folder this program will save figures in

mnist = tf.keras.datasets.mnist     # Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale data (min-max scaling)
x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test))/(np.max(x_test) - np.min(x_test))
#x_train = x_train/255
#x_test = x_test/255

# Print info about dataset
print("\nINFO:")
print("The MNIST dataset with images of handwritten digits")
print(f"Total number of observations: {len(y_train) + len(y_test)}")
print(f" - Of which {len(y_train)} are in the training set,")
print(f"   and {len(y_test)} are in the test set")
print(f"The shape of each image (X-value) is {x_train[0].shape} pixels\n")


# Visualize some the the numbers in a figure
sample_training_images = np.random.choice(len(y_train), 5)
sample_training_images = x_train[sample_training_images]
plotImages(sample_training_images, folder=folder, rgb = False)

"""
image_index = 111
plt.imshow(x_train[image_index], cmap='Greys')
plt.title(f"Example of handwritten {y_train[image_index]}")
save_fig("example_image", folder="mnist", extension="pdf")
save_fig("example_image", folder="mnist", extension="png")
plt.show()
"""

# Reshaping the array to 4-dims so that it can work with the Keras API
IMG_HEIGHT = x_train[0].shape[0]
IMG_WIDTH = x_train[0].shape[0]
x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
image_shape = (IMG_HEIGHT, IMG_WIDTH, 1)

# Set variables before preprocessing
batch_size = 128
epochs = 30

start_time = time.time()    # For recording the time the NN takes
print(f"Build NeuralNetwork model...")
# Build Convoluted Neural Network
model = tf.keras.Sequential([
    Conv2D(16, 1, padding='same', activation='relu', input_shape=image_shape),
    MaxPooling2D(),
    Conv2D(32, 1, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 1, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='sigmoid')
])
# Compile model
model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()     # print model summary

"""
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
"""

print("\nFit model on training data")
fit = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data = (x_test, y_test))

print("fit: ", fit.history) # The history holds record of loss and metric values during training
#print("\nEvaluate on test data: ")
#results = model.evaluate(x_test, y_test, batch_size=batch_size)
#print("test loss, test acc: ", results)

# Results
acc = fit.history['accuracy']           # Accuracy on training set
val_acc = fit.history['val_accuracy']   # Accuracy on validation (test) set
loss = fit.history['loss']              # Loss on training set
val_loss = fit.history['val_loss']      # Loss on validation (test) set
epochs_range = range(epochs)

# Print the CNN computation time
stop_time = time.time()
computing_time = stop_time - start_time
print(f"\n------------------\nCNN COMPUTING TIME: {computing_time}")

# Plot training/test accuracy and loss (error)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title('Training and Validation Loss')
save_fig("cnn_train_test_score", folder=folder, extension='pdf')
save_fig("cnn_train_test_score", folder=folder, extension='png')
plt.show()


