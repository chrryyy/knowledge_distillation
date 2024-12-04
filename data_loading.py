# Code adapted from https://www.kaggle.com/code/hojjatk/read-mnist-dataset 

import numpy as np
import struct
from array import array
import os
from PIL import Image

# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        

# Set file paths based on added MNIST Datasets
training_images_filepath = 'dataset/train-images.idx3-ubyte'
training_labels_filepath = 'dataset/train-labels.idx1-ubyte'
test_images_filepath = 'dataset/t10k-images.idx3-ubyte'
test_labels_filepath = 'dataset/t10k-labels.idx1-ubyte'

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# ----------------------------------------------------------
# FOR TESTING PURPOSES:

# # Save the first 5 images from the training set to a new folder
# first_5_images_folder = "first_5_images"

# # Create the folder if it doesn't exist
# if not os.path.exists(first_5_images_folder):
#     os.makedirs(first_5_images_folder)

# # Save the first 5 images and their labels
# for i in range(5):
#     img_array = np.array(x_train[i])  # Convert the list to a NumPy array
#     img = Image.fromarray(img_array)  # Convert the NumPy array to a PIL Image
#     img = img.convert("L")  # Ensure the image is in grayscale
#     img_path = os.path.join(first_5_images_folder, f"image_{i}_label_{y_train[i]}.png")
#     img.save(img_path)
#     print(f"Saved {img_path}")

# print(f"First 5 images saved to {first_5_images_folder}")
