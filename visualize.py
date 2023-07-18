import os
from glob import glob
import numpy as np
import importlib
from pathlib import Path
import shutil
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

viz = importlib.import_module("Annotation-Conversion.visualizer")

# Hyperparameters
width = 640
height = 640
square_size = 32

def ignore_files(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

# Paths
in_images_path = "./DAIS/"
out_images_path = "./out/"
tensors_path = "./tensors"

# Copy the tree structure
shutil.copytree(in_images_path, out_images_path, 
                ignore=ignore_files)

# List of all folders containing images
naloge = [os.listdir(in_images_path + x) for x in os.listdir(in_images_path)]

images = sorted([y for x in os.walk(in_images_path) for y in glob(os.path.join(x[0], '*.jpg'))])
tensors = sorted([y for x in os.walk(tensors_path) for y in glob(os.path.join(x[0], '*.pt'))])

for image_path, tensor_path in tqdm(zip(images, tensors), total=len(images)):
    image_name = os.path.basename(image_path)
    tensor = torch.load(tensor_path)

    # Initialize the visualizer
    visualizer = viz.Visualizer(image_path, width, height, square_size)
    
    visualizer.draw_grid()
    visualizer.draw_cartesian_predictors(tensor)
    visualizer.draw_cartesian_intersections(tensor)

    # Save the image
    out_image_path = out_images_path + '/'.join(image_path.split('/')[2:-1]) + '/'
    visualizer.save_image(out_path=out_image_path)











