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
    fig, axes = plt.subplots(1, len(images_arr), figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        if img.shape[-1] == 1:                  # If image is grayscale
            img = img.reshape(IMG_HEIGHT, IMG_WIDTH)    # Reshape
            ax.imshow(img, cmap='gray')
        elif rgb == False:
            ax.imshow(img, cmap = 'Greys')
        else:
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()

    if len(img) == 2:       # If grayscale
        save_fig("sample_images_(grayscale)", folder=folder, extension='pdf')
        save_fig("sample_images_(grayscale)", folder=folder, extension='png')
    else:
        save_fig("sample_images", folder=folder, extension='pdf')
        save_fig("sample_images", folder=folder, extension='png')
    if show == True:
        plt.show()
    else:
        plt.close()
