import cv2
import numpy as np

from Utils import *

# @TODO:
# - [ ] Use normalized DLT with RANSAC to estimate homography
# - [ x ] Use normalized 8-point algorithm with RANSAC to estimate fundamental matrix

# Checklist
# - [ x ] Initially, we just assume a non-planar scene and just use a fundamental matrix
#   - [ x ] Fundamental matrix estimation
#   - [ x ] Symmetric transfer error
#   - [ x ] RANSAC estimation scheme
#
# - [ ] Afterwards, work on pose estimation and triangulation
#   - [ ] Estimate the four motion hypotheses
#   - [ ] ...
#
# - [ ] Finally, implement bundle adjustment with g2o-python

class Initializer:
    def __init__(self, sigma, iterations):
        self.sigma      = sigma
        self.iterations = iterations

    def Initialize(self, initial_frame, current_frame, matches):
        # Part 1: Computation of (one) two models

        # Determine correspondences
        # a) initial_frame : query : src : 1
        # b) current_frame : train : dst : 2
        pts1 = []
        pts2 = []
        for match in matches:
            pts1.append(initial_frame.keypoints[match.queryIdx].pt)
            pts2.append(current_frame.keypoints[match.trainIdx].pt)
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        # Compute fundamental matrix via normalized 8-point algorithm and RANSAC
        F, score, inlier = Utils.EstimateFundamentalMatrixRANSAC(pts1, pts2, self.iterations, self.sigma, debug=True)
        Utils.ReconstructWithF(initial_frame.K, F, pts1, pts2, inlier)
