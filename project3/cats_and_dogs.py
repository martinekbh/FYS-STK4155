
import numpy as np
import matplotlib.pyplot as plt
import os
from own_code import *
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from skimage.color import rgb2gray


"""
SEE THIS ARTICLE
https://www.tensorflow.org/tutorials/images/classification
"""

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# This is the path to the data
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# Path to the training and the testing data
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
folder = "cats_and_dogs"

# Path to the training/test data divided into cats and dogs
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# Collect info about the dataset
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

# Print info
print("DATASET INFO:")
print('Total training cat images:', num_cats_tr)
print('Total training dog images:', num_dogs_tr)
print('Total validation cat images:', num_cats_val)
print('Total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("_______________________")

# Set variables before preprocessing
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
color_mode = 'rgb'
if color_mode == 'grayscale':
    img_layers = 1
elif color_mode == 'rgb':
    img_layers = 3
image_shape = (IMG_HEIGHT, IMG_WIDTH, img_layers)

# Use ImageDataGenerator to read images from disk and
# convert the images to into batches of tensors
print("\nGathering the images from the dataset...")
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode = color_mode,
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode = color_mode,
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)
sample_test_images, _ = next(test_data_gen)
print(sample_training_images.shape)
# Show some of the pictures from the training set
plotImages(sample_training_images[:5], folder="cats_and_dogs")

def fitCNN(batch_size = 128, epochs = 15):
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode = color_mode,
                                                           class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode = color_mode,
                                                              class_mode='binary')

    # Create CNN model and record the time it takes
    start_time = time.time()
    print("\nCreate Convoluted Neural Network model...")
    image_shape = (IMG_HEIGHT, IMG_WIDTH, img_layers)
    model = Sequential([
        Conv2D(32, img_layers, padding='same', activation='relu', input_shape=image_shape),
        MaxPooling2D(),
        Conv2D(32, img_layers, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, img_layers, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()     # print model summary

    # Train the model 
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )
    results = model.evaluate_generator(val_data_gen)
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
    pred = model.

    return model, history


def loopOverBatchSize(batch_sizes, epochs=15):
    print(f"\n\nLooping over batch size...")
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    fitted_models = []
    start_time = time.time()    # For recording the time the NN takes

    for batch_size in batch_sizes:
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode = color_mode,
                                                           class_mode='binary')

        val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode = color_mode,
                                                              class_mode='binary')
        model = Sequential([
            Conv2D(32, img_layers, padding='same', activation='relu', input_shape=image_shape),
            MaxPooling2D(),
            Conv2D(32, img_layers, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(32, img_layers, padding='same', activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        # Compile model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()     # print model summary

        # Train the model 
        fit = model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
        )


        print(f"\n--- Batch size {batch_size}...")

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
    save_fig("cnn_train_test_score_different_batchsizes_CD", folder=folder, extension='pdf')
    save_fig("cnn_train_test_score_different_batchsizes_CD", folder=folder, extension='png')
    plt.show()

    return fitted_models


if __name__ == '__main__':
    fitted_models = loopOverBatchSize([64, 32, 16, 8])
