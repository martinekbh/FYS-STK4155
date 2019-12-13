import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from shutil import copyfile
import os
import random


folder = 'data/original_data/train/'
images = []
categories = []

for file in os.listdir(folder):
    y = 0
    if file.startswith('cat'):
        y = 1

    image = load_img(folder + file, target_size=(150,150))
    image = img_to_array(image)

    images.append(image)
    categories.append(y)

images = np.asarray(images)
categories = np.asarray(categories)

print(f"Shape of image array: {images.shape}")
print(f"Shape of category array: {categories.shape}")
np.save(os.path.join('data/', "cats_and_dogs_all_images.npy"), images)
np.save(os.path.join('data/', "cats_and_dogs_all_categories.npy"), categories)

# Make folders for data
data_folder = 'data/sorted_data/'
subsets = ['train/', 'test/']

for foldr in subsets:
    os.makedirs(data_folder + foldr + 'cats/', exist_ok = True)
    os.makedirs(data_folder + foldr + 'dogs/', exist_ok = True)
# The percentage of the images we will use for testing
ratio_used_for_testing = 0.25   
# Sort the images into folders
source = folder

for image in os.listdir(source):
    source_path = source + '/' + image
    destination = data_folder + 'train/'

    if random.random() < ratio_used_for_testing:
        destination = data_folder + 'test/'
    if image.startswith('cat'): # Image is cat
        new_filepath = destination + 'cats/' + image
        copyfile(source_path, new_filepath)
    elif image.startswith('dog'):   # Image is dog
        new_filepath = destination + 'dogs/' + image
        copyfile(source_path, new_filepath)

# Make .npy files for cats and dogs sorted into training and test sets
i=0     # 0 is train, 1 is test
for foldr in subsets:
    if i==0:    # Training images
            n_images = round((1-ratio_used_for_testing)*2500)
    elif i==1:  # Test images
            n_images = round((ratio_used_for_testing*2500))

    k=0
    for category in ['dogs/', 'cats/']:
        images = []
        categories = []
        y = k
        
        for file, i in zip(os.listdir(data_folder + foldr + category), range(n_images)):
            image = load_img(data_folder+foldr+category+file, target_size=(150,150))
            image = img_to_array(image)

            images.append(image)
            categories.append(y)

        images = np.asarray(images)
        categories = np.asarray(categories)
        np.save(os.path.join('data/', category[:-1]+'_'+foldr[:-1]+'_'+"images.npy"), images)
        np.save(os.path.join('data/', category[:-1]+'_'+foldr[:-1]+'_'+"categories.npy"), categories)

        k+=1
    i+=1

