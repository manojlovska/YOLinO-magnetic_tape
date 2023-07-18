import os
from glob import glob
import numpy as np
import torch
import importlib
from pathlib import Path
import shutil
from tqdm import tqdm

conv = importlib.import_module("Annotation-Conversion.converter")

# Hyperparameters
width = 640
height = 640
square_size = 32

# For Cartesian and 1D points
# TODO: For Euler angles -> different function
def toTensor(squares):
    # To tensor
    img_coord = np.empty((int(width/square_size), int(height/square_size), squares.shape[-1]), dtype=np.float16)
    
    # We go through all squares horizontally and vertically
    for i in range(0, squares.shape[0]):
        for j in range(0, squares.shape[1]):
            square_coord = squares[i][j]

            if np.any(square_coord) is None:
                c = 0
                img_coord[i][j][:-1] = (squares.shape[-1]-1)*[0]
                img_coord[i][j][-1] = c

            else:
                c = 1
                img_coord[i][j][:-1] = square_coord[:-1]
                img_coord[i][j][-1] = c

    img_tensor = torch.from_numpy(img_coord)
    return img_tensor

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

# Paths
annotation_path = "./annotations.xml"
in_images_path = "./DAIS/"
out_tensors_path = "./tensors/"

# Copy the tree structure
shutil.copytree(in_images_path, out_tensors_path, 
                ignore=ignore_files)

# Initialize the converter
converter = conv.Converter(annotation_path, width, height, square_size)

# List of all folders containing images
naloge = [os.listdir(in_images_path + x) for x in os.listdir(in_images_path)]

# Recursevly find all images in in_image_path
images = [y for x in os.walk(in_images_path) for y in glob(os.path.join(x[0], '*.jpg'))]

# Go through every image and generate tensor for each one
for image_path in tqdm(images):
    image_name = os.path.basename(image_path)

    polylines = converter.get_polylines(image_name)
    squares = converter.to_cartesian(polylines)

    img_tensor = toTensor(squares)
    
    # Save tensors in tensor folder
    tensor_name = '.'.join(image_name.split('/')[-1].split('.')[:-1]) + '_tensor.pt'
    tensor_path = out_tensors_path + '/'.join(image_path.split('/')[2:-1]) + '/' + tensor_name

    torch.save(img_tensor, tensor_path)








