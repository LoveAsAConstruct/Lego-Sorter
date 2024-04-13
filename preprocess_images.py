import os
import cv2
import random
import numpy as np
from PIL import Image

from brick_data_struct import *

IMAGE_PATH = "archive\\LEGO brick images v1"

brickDict = {}

def get_image_info():
    pathdict = {}
    id = 0
    for directory in os.listdir(IMAGE_PATH):
        image_folder = f"{IMAGE_PATH}\\{directory}"
        folder_type = directory
        for image in os.listdir(image_folder):
            image_directory = f"{image_folder}\\{image}"
            pathdict.update({
                id: {
                    "id": id,
                    "path": image_directory,
                    "type": folder_type
                }
            })
            id += 1
    print(f"Obtained {len(pathdict)} image paths")
    return pathdict

brickdict = get_image_info() 

def write_original_images(brickdict, num_alts = 10):
    for id in brickdict:
        original_img = cv2.imread(brickdict[id]["path"], cv2.IMREAD_UNCHANGED)  # Load the image with alpha channel
        background_color = (255, 255, 255, 255)  # White background with full opacity
        background_img = np.full_like(original_img, background_color)

        # Extract the alpha channel from the original image
        alpha_channel = original_img[:, :, 3]

        # Create a mask for the transparent parts of the image
        mask = cv2.bitwise_not(alpha_channel)

        # Invert the mask to keep the non-transparent parts
        mask_inv = cv2.bitwise_not(mask)

        for i in range(0, num_alts):
            # Adjust brightness of the non-transparent parts of the original image
            brightness_factor = random.random() * 1.2  # Modify this value to adjust brightness
            bright_img = cv2.convertScaleAbs(original_img[:, :, :3], alpha=brightness_factor, beta=0)

            # Remove the alpha channel from the background image
            background_img_rgb = background_img[:, :, :3]

            # Create the processed image with the adjusted brightness and background
            processed_img = cv2.bitwise_and(bright_img, bright_img, mask=mask_inv) + cv2.bitwise_and(background_img_rgb, background_img_rgb, mask=mask)
            
            # Add noise to the processed image
            noise = np.random.normal(0, random.random()*20, processed_img.shape)
            noisy_img = np.clip(processed_img + noise, 0, 255).astype(np.uint8)
            output_path = os.path.join("dataset", f"processed_{id}_{i}.png")
            cv2.imwrite(output_path, noisy_img)

def sample_dataset():
    dataset_dir = "/path/to/your/dataset"