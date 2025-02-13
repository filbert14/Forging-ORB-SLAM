import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

class EDSLoader:

    FPS = 75

    # Calibration matrix
    K = np.array([[766.536025127154, 0                , 291.0503512057777],
                  [0               , 767.5749459126396, 227.4060484950132],
                  [0               , 0                , 1               ]])

    def __init__(self, dataset, sequence, fps=30):
        self.dataset  = dataset
        self.sequence = sequence
        self.fps      = fps
        self.images   = []

        # Load images
        images_path = os.path.join(self.dataset, self.sequence, "images")

        image_files = [image_file for
                       image_file in
                       sorted(os.listdir(images_path)) if
                       not image_file == "timestamps.txt"]

        step = int(np.ceil(EDSLoader.FPS / self.fps))
        image_files = image_files[::step]

        for image_file in tqdm(image_files, desc="Loading images"):
            self.images.append(cv2.imread(os.path.join(images_path, image_file), cv2.IMREAD_GRAYSCALE))
