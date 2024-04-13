import os

IMAGE_PATH = "archive\LEGO brick images v1"

brickDict = {}

def preprocess():
    for directory in os.listdir(IMAGE_PATH):
        image_folder = f"{IMAGE_PATH}\\{directory}"
        print(f"getting images in '{image_folder}'")

preprocess()