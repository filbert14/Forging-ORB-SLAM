import os

import cv2
from tqdm import tqdm

class Visualizer:
    def __init__(self):
        pass

    def save2D(self, images, save_dir):
        os.mkdir(save_dir)

        for i, image in enumerate(tqdm(images, desc="Saving images")):
            cv2.imwrite(os.path.join(save_dir, f"{i:06}.png"), image)