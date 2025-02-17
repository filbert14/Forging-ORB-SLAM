import random
import math

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
        # pts is an np.array of shape (N, D)
        return np.asarray([np.append(point, [1]) for point in pts])

    @staticmethod
    def ToEuclidean(pts):
        # pts is an np.array of shape (N, D)
        return np.asarray([np.array(point[:-1] / point[-1]) for point in pts])

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

    @staticmethod
    def EstimateFundamentalMatrixRANSAC(pts1, pts2, iterations, sigma, debug=False):
        N = pts1.shape[0]
        indices_all = range(N)

        best_F = None
        best_score = float('-inf')
        best_inlier = None

        for i in range(iterations):
            indices = random.sample(indices_all, 8)
            pts1_min_subset = pts1[indices]
            pts2_min_subset = pts2[indices]

            F = Utils.EstimateFundamentalMatrix(pts1_min_subset, pts2_min_subset)
            score, inlier = Utils.ComputeScoreFundamental(F, pts1, pts2, sigma)

            if score > best_score:
                best_F = F
                best_score = score
                best_inlier = inlier

            if debug:
                curr_num_inliers = int(inlier[inlier == 1].sum())
                best_num_inliers = int(best_inlier[best_inlier == 1].sum())
                print(f"iteration: {i + 1:03d} -- best_score: {best_score:.3f} -- num_inliers: {best_num_inliers:03d} -- score: {score:.3f} -- num_inliers: {curr_num_inliers:03d}")

        return best_F, best_score, best_inlier

    @staticmethod
    def DecomposeEssentialMatrix(E):
        # Decompose E using SVD
        U, S, VT = np.linalg.svd(E, full_matrices=True)

        # Ensure that U, V are valid rotations
        if np.linalg.det(U) < 0:
            U *= -1
            S *= -1

        if np.linalg.det(VT) < 0:
            VT *= -1
            S *= -1

        # Determine four different combinations of R and t
        W = np.zeros((3, 3))
        W[0, 1] = -1
        W[1, 0] = 1
        W[2, 2] = 1

        Rs = []
        ts = []

        t = U[:, 2]

        Rs.append(U @ W @ VT)
        ts.append(t)

        Rs.append(U @ W.T @ VT)
        ts.append(t)

        Rs.append(U @ W @ VT)
        ts.append(-t)

        Rs.append(U @ W.T @ VT)
        ts.append(-t)

        return Rs, ts

    @staticmethod
    def TriangulatePoints(P1, P2, pts1, pts2):
        def TriangulatePoint(P1, P2, point1, point2):
            # Constraints for point1
            x1, y1 = point1[0], point1[1]
            C1 = x1 * P1[2, :] - P1[0, :]
            C2 = y1 * P1[2, :] - P1[1, :]

            # Constraints for point2
            x2, y2 = point2[0], point2[1]
            C3 = x2 * P2[2, :] - P2[0, :]
            C4 = y2 * P2[2, :] - P2[1, :]

            # Create design matrix
            A = np.vstack((C1, C2, C3, C4))

            # Return minimizing right singular vector
            U, S, VT = np.linalg.svd(A, full_matrices=True)
            return VT[-1, :]

        N = pts1.shape[0]
        pts3D = np.array([TriangulatePoint(P1, P2, pts1[i], pts2[i]) for i in range(N)])

        return pts3D

    @staticmethod
    def CheckRT(K, R, t, pts1, pts2, inlier, sigma):
        # Construct projection matrix (initial frame) P1 = K[I | 0]
        P1 = np.hstack((K, np.zeros([3])[:, np.newaxis]))

        # Construct projection matrix (current frame) P2 = KR[I | -C] = K[R | -RC] = K[R | t]
        Rt = np.hstack((R, t[:, np.newaxis]))
        P2 = K @ Rt

        # Get optical center (initial frame) w.r.t. world coordinates
        O1 = np.zeros(3)

        # Get optical center (current frame) w.r.t. world coordinates
        O2 = -R.T @ t

        # Get the number of point correspondences
        N = inlier.shape[0]

        # Triangulate 3D points
        pts3D = Utils.ToEuclidean(Utils.TriangulatePoints(P1, P2, pts1, pts2))

        num_good   = 0
        good       = np.zeros(N)
        parallaxes = []

        # We go over matches
        for i in range(N):
            # If the point correspondence is not an inlier, we ignore it
            if not inlier[i]:
                continue

            # If the triangulated point is not located at a finite location, we ignore it
            point3D = pts3D[i]
            if not np.isfinite(point3D).all():
                continue

            # Compute parallax
            normal1 = point3D - O1
            distan1 = np.linalg.norm(normal1)

            normal2 = point3D - O2
            distan2 = np.linalg.norm(normal2)

            cos_parallax = np.dot(normal1, normal2) / (distan1 * distan2)

            # Under sufficient parallax, if the triangulated point has negative depth w.r.t. any camera, we ignore it
            point3DCamera1 = point3D
            point3DCamera2 = R @ point3D + t

            if cos_parallax < 0.99998 and point3DCamera1[2] <= 0:
                continue

            if cos_parallax < 0.99998 and point3DCamera2[2] <= 0:
                continue

            # If the reprojection error surpasses a given threshold for any camera, we ignore the point correspondence
            threshold = 4 * (sigma ** 2)

            point3D_hom = Utils.ToHomogeneous(np.array([point3D]))[0]

            point2DCamera1 = Utils.ToEuclidean(np.array([P1 @ point3D_hom]))[0]
            squared_error_1  = np.linalg.norm(point2DCamera1 - pts1[i]) ** 2
            if squared_error_1 > threshold:
                continue

            point2DCamera2 = Utils.ToEuclidean(np.array([P2 @ point3D_hom]))[0]
            squared_error_2  = np.linalg.norm(point2DCamera2 - pts2[i]) ** 2
            if squared_error_2 > threshold:
                continue

            # Indicates sufficient parallax
            if cos_parallax < 0.99998:
                good[i] = True

            # Store quantities
            num_good += 1
            parallaxes.append(cos_parallax)

        if num_good > 0:
            parallaxes.sort()
            index = min(50, len(parallaxes) - 1)
            ang_parallax = math.acos(parallaxes[index]) * (180/np.pi)
        else:
            ang_parallax = 0

        return pts3D, num_good, good, ang_parallax

    @staticmethod
    def ReconstructWithF(K, F, pts1, pts2, inlier, sigma):
        # Compute essential matrix
        E = K.T @ F @ K

        # Decompose essential matrix
        Rs, ts = Utils.DecomposeEssentialMatrix(E)

        # Reconstruct with all four hypotheses
        Utils.CheckRT(K, Rs[0], ts[0], pts1, pts2, inlier, sigma)
