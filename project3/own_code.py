import os
import numpy as np
import matplotlib.pyplot as plt


# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
#DATA_ID = "DataFiles/"

# Make these folders if they do not exist
if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
#if not os.path.exists(DATA_ID):
#    os.makedirs(DATA_ID)


def image_path(fig_id, folder=""):
    """ Function to create path for new figure """
    if not os.path.exists(os.path.join(FIGURE_ID, folder)):
        os.makedirs(os.path.join(FIGURE_ID, folder))
    return os.path.join(FIGURE_ID, folder, fig_id)

def save_fig(fig_id, folder="", extension='pdf'):
    """ Function for saving figure in correct folder """
    plt.savefig(image_path(fig_id, folder=folder) + '.' + extension, format=extension)

#def data_path(dat_id):
#    return os.path.join(DATA_ID, dat_id)


# Function for plotting a selected number of images from the training
# set in one figure
def plotImages(images_arr, folder= "", rgb=True, show=False):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(20,4))
    axes = axes.flatten()
    grayscale = False
    for img, ax in zip(images_arr, axes):
        if img.shape[-1] == 1:                  # If image is grayscale
            print("image is grayscale")
            grayscale=True
            IMG_HEIGHT = images_arr[0].shape[0]
            IMG_WIDTH = images_arr[0].shape[1]
            img = img.reshape(IMG_HEIGHT, IMG_WIDTH)    # Reshape
            ax.imshow(img, cmap='gray')
        elif rgb == False:
            ax.imshow(img, cmap = 'Greys')
        else:
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()

    if folder == "mnist":
        figure_name = "sample_images_MNIST"
    elif folder == "cats_and_dogs":
        figure_name = "sample_images_CD"
    else:
        figure_name = "sample_images"

    if grayscale:       # If grayscale
        save_fig(figure_name + "_(grayscale)", folder=folder, extension='pdf')
        save_fig(figure_name + "_(grayscale)", folder=folder, extension='png')
    else:
        save_fig(figure_name, folder=folder, extension='pdf')
        save_fig(figure_name, folder=folder, extension='png')
    if show == True:
        plt.show()
    else:
        plt.close()
