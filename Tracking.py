from enum import Enum

import cv2

class State(Enum):
    NO_IMAGES_YET   = 0
    NOT_INITIALIZED = 1
    INITIALIZING    = 2

class Tracking:
    def __init__(self, images):
        # Class variables
        self.images  = images
        self.img_idx = 0
        self.state   = State.NO_IMAGES_YET

    def FirstInitialization(self):
        print("Hello world!")

    def GrabImage(self):
        # Acquire next image
        image         = self.images[self.img_idx]
        self.img_idx += 1

        if self.state == State.NO_IMAGES_YET:
            self.state = State.NOT_INITIALIZED

        if self.state == State.NOT_INITIALIZED:
            self.FirstInitialization()

def main():
    tracking = Tracking([1, 2, 3, 4, 5])
    tracking.GrabImage()

if __name__ == "__main__":
    main()