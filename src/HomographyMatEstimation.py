import cv2 as cv
import numpy as np

from config import *


class HomographyMatrix:
    def __init__(self, methodOptimizer = cv.FM_RANSAC, ransacReprojThreshold = 2.0):
        ## /** 0(None), 4(cv.LMEDS), (8)RANSAC, 16(RHO) **/ ##
        self.methodOptimizer = methodOptimizer
        self.ransacReprojThreshold = ransacReprojThreshold


    def compute_HomographyMatrix (self, src_pts, dst_pts, image_A, image_B, matches):
        NB_Matching_Threshold = 4
        if len(src_pts) < NB_Matching_Threshold:
            return None, None

        # minVal, maxVal, _, _ = cv.minMaxLoc(src_pts)
        # self.ransacReprojThreshold = 0.004 * maxVal

        self.HomographyMat, maskInliers = cv.findHomography(src_pts, dst_pts, method = self.methodOptimizer, ransacReprojThreshold = self.ransacReprojThreshold)

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                 singlePointColor = None,
        #                 matchesMask = maskInliers, # draw only inliers
        #                 flags = 2)

        # goodMatchesImage = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints, matches, None, **draw_params)
        # cv.imshow("Homog_"+str(image_A.id)+"_"+str(image_B.id), goodMatchesImage)

        return maskInliers.reshape(-1)
            