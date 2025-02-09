import os
import sys

import cv2
from tqdm import tqdm

class EDSLoader:
    def __init__(self, dataset, sequence):
        self.dataset  = dataset
        self.sequence = sequence

        images_path = os.path.join(self.dataset, self.sequence, "images")

        image_files = [image_file for
                       image_file in
                       sorted(os.listdir(images_path)) if
                       not image_file == "timestamps.txt"]

        self.images = []
        for image_file in tqdm(image_files, desc="Loading images"):
            self.images.append(cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE))
