import cv2
import numpy as np

"""
# Some notes on the chi-squared test:

- If a random variable Z follows the standard normal distribution, then Z^2 has the chi^2 distribution with one degree of freedom (DOF)
- If Z1, Z2, ..., Zk are independent standard normal random variables, then Z1^2 + ... + Zk^2 has the chi^2 distribution with K DOFs
- We assume that the reprojection error is a random variable which has the normal distribution with mu = 0 and std = 1 (in pixels)

- If we normalize by std and square afterwards, we get a quantity which has a chi^2 distribution (with 1 DOF)
- Given the DOF, we can look at a chi-square distribution table for the significance level that we want
- For 1 DOF and a significance level of 0.05 ^= 5%, we can extract the critical value 3.841

- This means that the probability of the quantity taking a value between 0 and 3.841 is 95%
- Conversely, the probability of the quantity taking a value bigger than 3.841 is 0.05 ^= 5%
- Intuitively, we then consider the point an outlier if the corresponding quantity takes on a value bigger than 3.841,
  since the probability of its occurrence is only 0.05 ^= 5%
"""

class Utils:
    sig51DOF = 3.841
    sig52DOF = 5.991

    @staticmethod
    # pts is an np.array of shape (N, 2)
    def GetCondition2D(pts):
        centroid = pts.mean(axis=0)

        sx, sy = [], []
        for point in (pts - centroid):
            sx.append(np.abs(point[0]))
            sy.append(np.abs(point[1]))

        sx = np.mean(sx)
        sy = np.mean(sy)

        cx = centroid[0]
        cy = centroid[1]

        T = np.array([[(1/sx),      0, -cx/sx],
                      [     0, (1/sy), -cy/sy],
                      [     0,      0,     1]])

        return T

    @staticmethod
    def ToHomogeneous(pts):
        # pts is an np.array of shape (N, 2)
        pts_hom = []

        for point in pts:
            x, y = point[0], point[1]
            pts_hom.append(np.array([x, y, 1]))

        return np.asarray(pts_hom)

    @staticmethod
    def ToEuclidean(pts):
        # pts is an np.array of shape (N, 3)
        pts_euc = []

        for point in pts:
            x, y, w = point[0], point[1], point[2]

            x_norm = x/w
            y_norm = y/w

            pts_euc.append(np.array([x_norm, y_norm]))

        return np.asarray(pts_euc)

    @staticmethod
    def ApplyHomography2D(pts, H):
        # pts is an np.array of shape (N, 2)
        # H is a 2D homography of shape (3, 3)
        pts_hom = Utils.ToHomogeneous(pts)

        pts_trf = np.array([H @ point for point in pts_hom])

        return Utils.ToEuclidean(pts_trf)

    @staticmethod
    def EstimateFundamentalMatrix(pts1, pts2):
        T1 = Utils.GetCondition2D(pts1)
        T2 = Utils.GetCondition2D(pts2)

        pts1_cond = Utils.ApplyHomography2D(pts1, T1)
        pts2_cond = Utils.ApplyHomography2D(pts2, T2)

        F, _ = cv2.findFundamentalMat(pts1_cond, pts2_cond, cv2.FM_8POINT)
        F_decond = T2.T @ F @ T1

        return F_decond/F_decond[2, 2]

    @staticmethod
    def ComputeScoreFundamental(F, pts1, pts2, sigma):
        N = pts1.shape[0]

        pts1_hom = Utils.ToHomogeneous(pts1)
        pts2_hom = Utils.ToHomogeneous(pts2)

        # Assume all correspondences are inliers
        score  = 0
        inlier = np.ones(N)

        for i in range(N):
            point1_hom = pts1_hom[i]
            point2_hom = pts2_hom[i]

            # Compute reprojection error in the second image
            line2 = F @ point1_hom
            a2, b2, c2 = line2[0], line2[1], line2[2]
            
            dist2 = np.dot(point2_hom, line2) / np.linalg.norm(np.array([a2, b2]))

            # Compute reprojection error in the first image
            line1 = F.T @ point2_hom
            a1, b1, c1 = line1[0], line1[1], line1[2]
            
            dist1 = np.dot(point1_hom, line1) / np.linalg.norm(np.array([a1, b1]))

            # Compute score
            chi_square_1 = (dist1/sigma) ** 2
            chi_square_2 = (dist2/sigma) ** 2

            if chi_square_1 > Utils.sig51DOF:
                inlier[i] = False
            else:
                score += Utils.sig52DOF - chi_square_1

            if chi_square_2 > Utils.sig51DOF:
                inlier[i] = False
            else:
                score += Utils.sig52DOF - chi_square_2

        return score, inlier
