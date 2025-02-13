import cv2
import numpy as np

class Utils:
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
