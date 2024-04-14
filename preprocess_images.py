import os
import cv2
import csv
import random
import numpy as np
from PIL import Image

from brick_data_struct import *

IMAGE_PATH = "archive\\LEGO brick images v1"


def get_image_info():
    pathdict = []
    id = 0
    index = 0
    for directory in os.listdir(IMAGE_PATH):
        image_folder = f"{IMAGE_PATH}\\{directory}"
        folder_type = directory
        for image in os.listdir(image_folder):
            image_directory = f"{image_folder}\\{image}"
            pathdict.append(
                {
                    "id": id,
                    "path": image_directory,
                    "type": folder_type
                }
            )
            index += 1
        id += 1
    print(f"Obtained {len(pathdict)} image paths with {id} UIDs")
    return pathdict

def write_original_images(brickdict, num_alts = 10):
    for i, index in enumerate(brickdict):
        original_img = cv2.imread(index["path"], cv2.IMREAD_UNCHANGED)  # Load the image with alpha channel
        background_color = (255, 255, 255, 255)  # White background with full opacity
        background_img = np.full_like(original_img, background_color)

        # Extract the alpha channel from the original image
        alpha_channel = original_img[:, :, 3]

        # Create a mask for the transparent parts of the image
        mask = cv2.bitwise_not(alpha_channel)

        # Invert the mask to keep the non-transparent parts
        mask_inv = cv2.bitwise_not(mask)

        for j in range(0, num_alts):
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
            output_path = os.path.join("data/raw_dataset", f"processed_{index['id']}_{i}_{j}.png")
            cv2.imwrite(output_path, noisy_img)

def sample_dataset(width, height):
    dataset_dir = "data/raw_dataset"
    resized_dir = "data/sampled_data"
    for image_path in os.listdir(dataset_dir):
        img = Image.open(os.path.join(dataset_dir, image_path))
        resized_image = img.resize((width,height))
        resized_image.save(os.path.join(resized_dir, image_path))
    print("Resample complete")

def map_dataset():
    resized_dir = "data/raw_dataset"
    csv_file = "data/classification.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID, Filename, Classification"])
        id = 0
        for image_name in os.listdir(resized_dir):
            if image_name.endswith(".png"):
                class_id = image_name.split("_")[1]
                writer.writerow([id, image_name, class_id])
                id += 1


def store_dicts_to_csv(data, filename):
    """
    Stores a list of dictionaries to a CSV file.

    Parameters:
        data (list of dict): The list of dictionaries to store. Each dictionary should have the same structure.
        filename (str): The path to the CSV file where data will be stored.
    """
    if not data:
        print("The provided data list is empty.")
        return

    # Assuming all dictionaries have the same keys, use the keys from the first dictionary for fieldnames
    fieldnames = data[0].keys()

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write data rows
        for item in data:
            writer.writerow(item)

    print(f"Data has been successfully written to {filename}")


brickdict = get_image_info() 
store_dicts_to_csv(brickdict, "data/brickdict.csv")
write_original_images(brickdict, 4)
sample_dataset(100, 100)
map_dataset()