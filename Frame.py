class Frame:
    def __init__(self, image, keypoints, descriptors):
        # Original image
        self.image       = image

        # List of keypoints
        self.keypoints   = keypoints

        # List of descriptors
        self.descriptors = descriptors

        # Number of keypoints
        self.N           = len(keypoints)
