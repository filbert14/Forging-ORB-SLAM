import cv2
import numpy as np

# @TODO:
# - [ ] Use normalized DLT with RANSAC to estimate homography
# - [ ] Use normalized 8-point algorithm with RANSAC to estimate fundamental matrix

# Checklist
# - [ ] Initially, we just assume a non-planar scene and just use a fundamental matrix
#   - [ ] Fundamental matrix estimation
#   - [ ] Symmetric transfer error
#   - [ ] RANSAC estimation scheme
#   - [ ] Motion and structure reconstruction
# - [ ] Then, implement pose estimation
# - [ ] Then, implement bundle adjustment

class Initializer:
    def __init__(self, iterations):
        self.iterations = iterations

    def Initialize(self, initial_frame, current_frame, matches):
        # Part 1: Computation of (one) two models

        # initial_frame : query : src : 1
        # current_frame : train : dst : 2
        pts1 = []
        pts2 = []
        for match in matches:
            pts1.append(initial_frame.keypoints[match.queryIdx].pt)
            pts2.append(current_frame.keypoints[match.trainIdx].pt)
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        # Compute a homography
        H = cv2.findHomography(pts1, pts2, cv2.RANSAC, maxIters=self.iterations)

        # Compute a fundamental matrix
        F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, maxIters=self.iterations)
