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

# Print info about dataset
print("\nINFO:")
print("The MNIST dataset with images of handwritten digits")
print(f"Total number of observations: {len(y_train) + len(y_test)}")
print(f" - Of which {len(y_train)} are in the training set,")
print(f"   and {len(y_test)} are in the test set")
print(f"The shape of each image (X-value) is {x_train[0].shape} pixels\n")

def sample_images():
    # Visualize some the the numbers in a figure
    sample_training_images = np.random.choice(len(y_train), 5)
    sample_training_images = x_train[sample_training_images]
    plotImages(sample_training_images, folder=folder, rgb = False)


# Reshaping the array to 4-dims so that it can work with the Keras API
IMG_HEIGHT = x_train[0].shape[0]
IMG_WIDTH = x_train[0].shape[0]
x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
image_shape = (IMG_HEIGHT, IMG_WIDTH, 1)

def fitCNN(batch_size=32, epochs=18):
    """ Function that classifies the MNIST data with a simple CNN """

    start_time = time.time()    # For recording the time the NN takes
    print(f"Build NeuralNetwork model...")
    # Build Convoluted Neural Network
    model = tf.keras.Sequential([
        Conv2D(28, kernel_size = (3,3), padding='same', activation='relu', input_shape=image_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32, 1, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        #Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    # Compile model
    model.compile(optimizer='adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()     # print model summary

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

    print(f"\nModel performance:")
    model.evaluate(x_test, y_test)

    # Print the CNN computation time
    stop_time = time.time()
    computing_time = stop_time - start_time
    print(f"\n------------------\nCNN COMPUTING TIME: {computing_time}")

    # Plot training/test accuracy and loss (error)
    plt.figure(figsize=(10, 8))
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
    save_fig("cnn_train_test_score_dropout_MNIST", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_dropout_MNIST", folder=folder, extension='png')
    plt.show()

    # Make confusion matrix:
    pred = model.predict_classes(x_test)
    print(pred[:10])
    print(y_test[:10])
    con_matr = confusionMatrix(pred, y_test, 
            figurename="cnn_confusion_matrix_dropout_MNIST", folder=folder, show=True)

    # Display some of the misclassified images
    imgShape = x_test.shape
    plotMisclassifiedImages(y_test, x_test.reshape(imgShape[0], 28,28), pred, 
                figurename="sample_misclassified_images_mnist", folder = "mnist", rgb=False)                               
            #x_test, pred, 
            #figurename="sample_misclassified_images_dropout_mnist", folder=folder, rgb=False, show=True)

    return model, fit

def loopOverBatchSize(batch_sizes, epochs=10):
    """ Function that evaluates and plots the performance of CNNs (on the MNIST data) 
        with different batch sizes """

    print("\n\n Looping over batch size...")
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    fitted_models = []
    start_time = time.time()    # For recording the time the NN takes

    for batch_size in batch_sizes:
        print(f"Build NeuralNetwork model...")
        # Build Convoluted Neural Network
        model = tf.keras.Sequential([
            Conv2D(28, 1, padding='same', activation='relu', input_shape=image_shape),
            MaxPooling2D(),
            Conv2D(32, 1, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 1, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])
        # Compile model
        model.compile(optimizer='adam',
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        # Fit model on training data
        fit = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data = (x_test, y_test))

        # Results
        fitted_models.append([model, fit])
        acc.append(fit.history['accuracy'])         # Accuracy on training set
        val_acc.append(fit.history['val_accuracy']) # Accuracy on validation (test) set
        loss.append(fit.history['loss'])            # Loss on training set
        val_loss.append(fit.history['val_loss'])    # Loss on validation (test) set
    epochs_range = range(epochs)

    # Print the CNN computation time
    stop_time = time.time()
    computing_time = stop_time - start_time
    print(f"\n------------------\nCOMPUTING TIME: {computing_time}")

    # Plot training/test accuracy and loss (error)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'darkgreen', 'orange']   # List of colors
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    for i in range(len(fitted_models)):
        plt.plot(epochs_range, acc[i], c = colors[i], linestyle='--',
                label=f'Batch size {batch_sizes[i]} (train)')
        plt.plot(epochs_range, val_acc[i], c = colors[i], linestyle='-',
                label=f'Batch size {batch_sizes[i]} (test)')
    plt.legend(loc='lower right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    for i in range(len(fitted_models)):
        plt.plot(epochs_range, loss[i], c = colors[i], linestyle='--',
                label=f'Batch size {batch_sizes[i]} (train)')
        plt.plot(epochs_range, val_loss[i], c = colors[i], linestyle='-',
                label=f'Batch size {batch_sizes[i]} (test)')
    plt.legend(loc='upper right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title('Training and Validation Loss')
    save_fig("cnn_train_test_score_different_batchsizes_MNIST", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_different_batchsizes_MNIST", folder=folder, extension='png')
    plt.show()

    return fitted_models

def loopOverLayers(num_layers, batch_size=64, epochs=30):
    """ Function that evaluates and plots the performance of CNNs with different 
        number of hidden Conv2D and MaxPoolong2D layers. """

    print("\n\n Looping over the number of layers...")
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    fitted_models = []
    start_time = time.time()    # For recording the time the NN takes

    for l in range(num_layers):
        print(f"\n--- Build NeuralNetwork model with {l} hidden conv2D layers")
        # Input layer
        model = tf.keras.Sequential()
        model.add(Conv2D(32, 1, padding='same', activation='relu', input_shape=image_shape))
        model.add(MaxPooling2D())

        # Add hidden layers
        for k in range(l):
            model.add(tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D())

        # Output layers 
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # Compile model
        model.compile(optimizer='adam',
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        # Fit model on training data
        fit = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data = (x_test, y_test))
        # Results
        fitted_models.append([model, fit])
        acc.append(fit.history['accuracy'])         # Accuracy on training set
        val_acc.append(fit.history['val_accuracy']) # Accuracy on validation (test) set
        loss.append(fit.history['loss'])            # Loss on training set
        val_loss.append(fit.history['val_loss'])    # Loss on validation (test) set
    epochs_range = range(epochs)
    layers_range = range(num_layers)

    # Print the CNN computation time
    stop_time = time.time()
    computing_time = stop_time - start_time
    print(f"\n------------------\nCOMPUTING TIME: {computing_time}")

    # Plot training/test accuracy and loss (error)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'darkgreen', 'orange']   # List of colors
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    for i in range(num_layers):
        plt.plot(epochs_range, acc[i], c = colors[i], linestyle='--',
                label=f'{layers_range[i]} hidden layers (train)')
        plt.plot(epochs_range, val_acc[i], c = colors[i], linestyle='-',
                label=f'{layers_range[i]} hidden layers (test)')
    plt.legend(loc='lower right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    for i in range(num_layers):
        plt.plot(epochs_range, loss[i], c = colors[i], linestyle='--',
                label=f'{layers_range[i]} hidden layers (train)')
        plt.plot(epochs_range, val_loss[i], c = colors[i], linestyle='-',
                label=f'{layers_range[i]} hidden layers (test)')
    plt.legend(loc='upper right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    save_fig("cnn_train_test_score_different_numlayers_MNIST", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_different_numlayers_MNIST", folder=folder, extension='png')
    plt.show()

    return fitted_models
   

if __name__ == '__main__':

    fitCNN()
    #loopOverBatchSize(batch_sizes = [128, 96, 64, 32, 16], epochs = 30)
    #loopOverLayers(3)

