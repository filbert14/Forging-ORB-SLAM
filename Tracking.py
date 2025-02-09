from enum import Enum

import cv2

class State(Enum):
    NO_IMAGES_YET   = 0
    NOT_INITIALIZED = 1
    INITIALIZING    = 2

class Tracking:
    def __init__(self, images, settings):
        # Class variables
        self.images   = images
        self.settings = settings

        self.img_idx  = 0
        self.state    = State.NO_IMAGES_YET

        # ORB
        nfeatures     = settings["ORBextractor.nFeatures"]
        scaleFactor   = settings["ORBextractor.scaleFactor"]
        nlevels       = settings["ORBextractor.nLevels"]
        fastThreshold = settings["ORBextractor.fastTh"]
        scoreType     = settings["ORBextractor.nScoreType"]

        self.orb = cv2.ORB_create(
            nfeatures     = nfeatures,
            scaleFactor   = scaleFactor,
            nlevels       = nlevels,
            fastThreshold = fastThreshold,
            scoreType     = scoreType,
        )

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
