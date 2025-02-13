import cv2

class Frame:
    def __init__(self, image, keypoints, descriptors, K):
        # Original image
        self.image       = image.copy()

        # List of keypoints
        self.keypoints   = [Frame.CopyKeyPoint(keypoint) for keypoint in keypoints]

        # List of descriptors
        self.descriptors = descriptors.copy()

        # Number of keypoints
        self.N           = len(keypoints)

        # Calibration matrix
        self.K           = K.copy()

    @staticmethod
    def CopyKeyPoint(keypoint):
        return cv2.KeyPoint(x        = keypoint.pt[0],
                            y        = keypoint.pt[1],
                            size     = keypoint.size,
                            angle    = keypoint.angle,
                            response = keypoint.response,
                            octave   = keypoint.octave,
                            class_id = keypoint.class_id)
