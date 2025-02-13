from enum import Enum

import cv2

from Frame import Frame
from Matcher import Matcher
from Initializer import Initializer

class State(Enum):
    NO_IMAGES_YET   = 0
    NOT_INITIALIZED = 1
    INITIALIZING    = 2

class Tracking:
    def __init__(self, images, K, settings):
        # Class variables
        self.images   = images
        self.K        = K
        self.settings = settings

        self.img_idx  = 0
        self.state    = State.NO_IMAGES_YET

        self.current_frame = None
        self.initial_frame = None

        # ORB
        nfeatures     = self.settings["ORBextractor.nFeatures"]
        scaleFactor   = self.settings["ORBextractor.scaleFactor"]
        nlevels       = self.settings["ORBextractor.nLevels"]
        fastThreshold = self.settings["ORBextractor.fastTh"]
        scoreType     = self.settings["ORBextractor.nScoreType"]

        self.orb = cv2.ORB_create(
            nfeatures     = nfeatures,
            scaleFactor   = scaleFactor,
            nlevels       = nlevels,
            fastThreshold = fastThreshold,
            scoreType     = scoreType,
        )

    def FirstInitialization(self):
        if self.current_frame.N > 100:
            self.initial_frame = Frame(self.current_frame.image, self.current_frame.keypoints, self.current_frame.descriptors, self.K)
            self.state = State.INITIALIZING

    def Initialize(self):
        if self.current_frame.N <= 100:
            self.state = State.NOT_INITIALIZED
            return

        matcher = Matcher()
        matches = matcher.SearchForInitialization(self.initial_frame, self.current_frame)

        if len(matches) < 100:
            self.state = State.NOT_INITIALIZED
            return

        initializer = Initializer(1.0, 200)
        initializer.Initialize(self.initial_frame, self.current_frame, matches)

    def GrabImage(self):
        # Acquire next image
        image         = self.images[self.img_idx]
        self.img_idx += 1

        # Compute keypoints and descriptors for the current frame
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        self.current_frame     = Frame(image, keypoints, descriptors, self.K)

        if self.state == State.NO_IMAGES_YET:
            self.state = State.NOT_INITIALIZED

        if self.state == State.NOT_INITIALIZED:
            self.FirstInitialization()

        elif self.state == State.INITIALIZING:
            self.Initialize()
