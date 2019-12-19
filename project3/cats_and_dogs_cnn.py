# General imports
import numpy as np
import os
from own_code import *
import matplotlib.pyplot as plt
import time

# Import ML methods
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from skimage.color import rgb2gray


folder = "cats_and_dogs"

# load data
# if these .npy files are not present, you must run the file
# cats_and_dogs_generate_data.py first. It will create the files
print("Loading data (cats and dogs)...")
x_train_cats = np.load('data/cats_train_images.npy')
y_train_cats = np.load('data/cats_train_categories.npy')
x_train_dogs = np.load('data/dogs_train_images.npy')
y_train_dogs = np.load('data/dogs_train_categories.npy')
x_test_cats = np.load('data/cats_test_images.npy')
y_test_cats = np.load('data/cats_test_categories.npy')
x_test_dogs = np.load('data/dogs_test_images.npy')
y_test_dogs = np.load('data/dogs_test_categories.npy')
print("Data loaded!\n")

x_train = np.concatenate((x_train_cats, x_train_dogs), axis=0)
x_test = np.concatenate((x_test_cats, x_test_dogs), axis=0)
y_train = np.concatenate((y_train_cats, y_train_dogs), axis=None)
y_test = np.concatenate((y_test_cats, y_test_dogs), axis=None)

# Scale data (min-max scaling)
x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test))/(np.max(x_test) - np.min(x_test))


# Shuffle the arrays so cats and dogs are not sorted
shuffled_indexes = np.random.permutation(len(x_train))
x_train = x_train[shuffled_indexes]
y_train = y_train[shuffled_indexes]
shuffled_indexes = np.random.permutation(len(x_test))
x_test = x_test[shuffled_indexes]
y_test = y_test[shuffled_indexes]

# data shape parameters
print(x_train.shape)
image_shape = x_train.shape[1:]
IMG_HEIGHT = image_shape[0]
IMG_WIDTH = image_shape[1]

print("\nINFO:")
print("The cats and dogs dataset")
print(f"Total number of observations: {len(y_train) + len(y_test)}")
print(f" - Of which {len(y_test)} are in the test set,")
print(f"   and {len(y_train)} are in the training set.")
print(f"Shape of the images: {IMG_HEIGHT} x {IMG_WIDTH}")
print(f"The shape of each image is {image_shape}")

# Plot a sample of the images
sample_indexes = np.random.choice(range(len(x_train)), size=5)
sample_images = x_train[sample_indexes]
#sample_images = np.random.choice(x_train, size=(5,IMG_HEIGHT, IMG_WIDTH))
plotImages(sample_images, folder=folder)
print(y_train[sample_indexes])

def fitCNN(batch_size=64, epochs=15):
    """ Fits a simple CNN on the dataset """
    print("\nFit CNN with batch size {batch_size} and {epochs} epochs")

    start_time = time.time()
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', 
                input_shape=image_shape), #kernel_initializer = 'he_uniform'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),#, kernel_initializer='he_uniform'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()     # print model summary

    # Fit the model on training data
    history = model.fit(x_train, y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_data = (x_test, y_test))
    stop_time = time.time()
    computing_time = stop_time - start_time
    print(f"\n------------------\nCNN COMPUTING TIME: {computing_time}")

    #return model, history

#def evaluateModel(model, history, save_id=""):
    """ Evaluates a given model and produces plots """
    results = model.evaluate(x_test, y_test)
    print(results)

    # Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    
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
    save_fig("cnn_train_test_score_CD", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_CD", folder=folder, extension='png')
    plt.show()

    # Make predictions of the test/validation images
    pred = model.predict_classes(x_test)
    print(pred[:10])
    print(y_test[:10])

    # Make confusion matrix
    con_matr = confusionMatrix(pred, y_test, 
            figurename="cnn_confusion_matrix_CD", folder=folder, show=True)

    # Display some of the misclassified images
    plotMisclassifiedImages(y_test, x_test, pred.ravel(), 
            figurename="sample_misclassified_images_CD", folder=folder, rgb=True, show=True)

    return pred

def loopOverLayers(num_layers, batch_size=64, epochs=30, start_num_layers=0):
    """ Function that evaluates and plots the performance of CNNs with different
    """

    print("\n\n Looping over the number of layers...")
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    fitted_models = []
    start_time = time.time()    # For recording the time the NN takes

    for l in range(start_num_layers, num_layers):
        print(f"\n--- Build NeuralNetwork model with {l} hidden conv2D layers")
        # Input layer
        model = tf.keras.Sequential()
        model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=image_shape))
        model.add(MaxPooling2D((2,2)))

        # Add hidden layers
        for k in range(l):
            model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2,2)))

        # Output layers 
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      #loss = tf.keras.losses.SparseCategoricalCrossentropy(),
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
    layers_range = range(start_num_layers, num_layers)

    # Print the CNN computation time
    stop_time = time.time()
    computing_time = stop_time - start_time
    print(f"\n------------------\nCOMPUTING TIME: {computing_time}")

    # Print best results:
    max_valacc_ind = [max(a) for a in val_acc].index(np.max(val_acc))
    max_acc_ind = [max(a) for a in acc].index(np.max(acc))
    min_valloss_ind = [min(a) for a in val_loss].index(np.min(val_loss))
    min_loss_ind = [min(a) for a in loss].index(np.min(loss))
    print(f"Max test accuracy: {np.max(val_acc)} with {start_num_layers + max_valacc_ind} hidden conv2d layers")
    print(f"Max train accuracy: {np.max(acc)} with  {start_num_layers + max_acc_ind} hidden conv2d layers")
    print(f"Min test loss: {np.min(val_loss)} with {start_num_layers + min_valloss_ind} hidden conv2d layers")
    print(f"Min train loss: {np.min(loss)} with {start_num_layers + min_loss_ind} hidden conv2d layers")

    print("\a") # Alert the programmer that the function is almost done

    # Plot training/test accuracy and loss (error)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'darkgreen', 'orange']   # List of colors
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    for i in range(len(acc)):
        plt.plot(epochs_range, acc[i], c = colors[i], linestyle='--',
                label=f'{layers_range[i]} convolution layers (train)')
        plt.plot(epochs_range, val_acc[i], c = colors[i], linestyle='-',
                label=f'{layers_range[i]} convolution layers (test)')
    plt.legend(loc='lower right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    for i in range(len(loss)):
        plt.plot(epochs_range, loss[i], c = colors[i], linestyle='--',
                label=f'{layers_range[i]} convolution layers (train)')
        plt.plot(epochs_range, val_loss[i], c = colors[i], linestyle='-',
                label=f'{layers_range[i]} convolution layers (test)')
    plt.legend(loc='upper right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    save_fig("cnn_train_test_score_different_numlayers_CD", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_different_numlayers_CD", folder=folder, extension='png')
    plt.show()

    return fitted_models
   

def loopOverBatchsize(batch_sizes, epochs=35):
    """ Function that evaluates and plots the performance of CNNs (on the cats and dogs data)
        with different batch sizes """

    print("\n\n Looping over the batch size...")
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    fitted_models = []
    start_time = time.time()    # For recording the time the NN takes


    for batch_size in batch_sizes:
        print(f"Build NeuralNetwork model with batch size {batch_size}...")
        model = tf.keras.Sequential()
        model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=image_shape))
        model.add(MaxPooling2D((2,2)))

        # Add hidden layers
        for k in range(5):  # add 5 hidden convolution layers
            model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2,2)))

        # Output layers 
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
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
    print(f"\n------------------\nCOMPUTING TIME: {computing_time}\n")


    # Print results of best model:
    max_valacc_ind = [max(a) for a in val_acc].index(np.max(val_acc))
    max_acc_ind = [max(a) for a in acc].index(np.max(acc))
    min_valloss_ind = [min(a) for a in val_loss].index(np.min(val_loss))
    min_loss_ind = [min(a) for a in loss].index(np.min(loss))
    print(f"Max test accuracy: {np.max(val_acc)} with batch size {batch_sizes[max_valacc_ind]}")
    print(f"Max train accuracy: {np.max(acc)} with  batch size {batch_sizes[max_acc_ind]}")
    print(f"Min test loss: {np.min(val_loss)} with batch size {batch_sizes[min_valloss_ind]}")
    print(f"Min train loss: {np.min(loss)} with batch size {batch_sizes[min_loss_ind]}")

    print("\a") # Alert the programmer that the function is almost done

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
    save_fig("cnn_train_test_score_different_batchsizes_CD", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_different_batchsizes_CD", folder=folder, extension='png')
    plt.show()

    return fitted_models

def fitBestCNN(batch_size=40, epochs=30):
    """ Fits a CNN with 5 hidden convolution layers, batch size 40, 
        and 30 epochs on the dataset """
    print("\nFit CNN with batch size {batch_size} and {epochs} epochs")
    start_time = time.time()

    # Make CNN model
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D((2,2)))

    # Add hidden layers
    for k in range(5):  # add 5 hidden convolution layers
        model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))

    # Output layers 
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()     # print model summary

    # Fit the model on training data
    history = model.fit(x_train, y_train,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_data = (x_test, y_test))
    stop_time = time.time()
    computing_time = stop_time - start_time
    print(f"\n------------------\nCNN COMPUTING TIME: {computing_time}")

    """ Evaluates a given model and produces plots """
    results = model.evaluate(x_test, y_test)
    print(results)

    # Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    
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
    save_fig("cnn_train_test_score_best_model_CD", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_best_model_CD", folder=folder, extension='png')
    plt.show()

    # Make predictions of the test/validation images
    pred = model.predict_classes(x_test)
    print(pred[:10])
    print(y_test[:10])

    # Make confusion matrix
    con_matr = confusionMatrix(pred, y_test, 
            figurename="cnn_confusion_matrix_best_model_CD", folder=folder, show=True)

    # Display some of the misclassified images
    plotMisclassifiedImages(y_test, x_test, pred.ravel(), 
            figurename="sample_misclassified_images_best_model_CD", folder=folder, rgb=True, show=True)

    return pred


if __name__ == '__main__':
    #model, history = fitCNN()
    #pred = evaluateModel(model, history)
    #fitCNN()
    #fitted_models_layers = loopOverLayers(7)
    #fitted_models_batchsize = loopOverBatchsize([30, 40, 50, 60])
    fitBestCNN(epochs=35)

