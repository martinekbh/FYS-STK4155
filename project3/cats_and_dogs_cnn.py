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
from skimage.color import rgb2gray


folder = "cats_and_dogs"

# load data
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
print(x_train.shape)

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
image_shape = x_train.shape[1:]
IMG_HEIGHT = image_shape[0]
IMG_WIDTH = image_shape[1]
print(f"Shape of the images: {IMG_HEIGHT} x {IMG_WIDTH}")

plt.imshow(x_train[0])
plt.show()
print(y_train[0])

# Plot a sample of the images
sample_indexes = np.random.choice(range(len(x_train)), size=5)
print(sample_indexes)
sample_images = x_train[sample_indexes]
print(sample_images)
#sample_images = np.random.choice(x_train, size=(5,IMG_HEIGHT, IMG_WIDTH))
plotImages(sample_images, folder=folder, show=True)
print(y_train[sample_indexes])

def fitCNN(batch_size=64, epochs=15):
    """ Fits a simple CNN on the dataset """
    print("\nFit CNN with batch size {batch_size} and {epochs} epochs")
    model = Sequential()
    model = Sequential([
        Conv2D(32, (3,3), img_layers, padding='same', activation='relu', 
                input_shape=image_shape, kernel_initializer = 'he_uniform'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
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

def evaluateModel(model, history, save_id=""):
    """ Evaluates a given model and produces plots """
    results = model.evaluate(x_test, y_test)
    print(results)

    # Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

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
    save_fig("cnn_train_test_score_CD", folder="cats_and_dogs", extension='pdf')
    save_fig("cnn_train_test_score_CD", folder="cats_and_dogs", extension='png')
    plt.show()

    # Make predictions of the test/validation images
    pred = model.predict_classes(x_test)

    # Make confusion matrix
    con_matr = confusionMatrix(pred, y_test, 
            figurename="cnn_confusion_matrix_dropout_MNIST", folder=folder, show=True)

    # Display some of the misclassified images
    plotMisclassifiedImages(y_test, x_test, pred, 
            figurename="sample_misclassified_images_dropout_mnist", folder=folder, rgb=False, show=True)

    return pred



if __name__ == '__main__':
    model, history = fitCNN()
    pred = evaluateModel(model, history)
