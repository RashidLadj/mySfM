import cv2 as cv
import numpy as np

from config import *


class HomographyMatrix:
    def __init__(self, methodOptimizer = cv.FM_RANSAC, ransacReprojThreshold = 2.0):
        ## /** 0(None), 4(cv.LMEDS), (8)RANSAC, 16(RHO) **/ ##
        self.methodOptimizer = methodOptimizer
        self.ransacReprojThreshold = ransacReprojThreshold


    #####################################################################
    ## NOTE: Pose is calculated from dst to src so it's the same for F
    # 
    #  ##
    #####################################################################
    def compute_HomographyMatrix (self, matching):
        NB_Matching_Threshold = 4
        if len(matching.prec_pts) < NB_Matching_Threshold:
            return None, None

        self.HomographyMat, maskInliers = cv.findHomography(matching.curr_pts, matching.prec_pts, method = self.methodOptimizer, ransacReprojThreshold = self.ransacReprojThreshold)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = maskInliers, # draw only inliers
                        flags = 2)
        goodMatchesImage = cv.drawMatches(matching.image_A.imageRGB, matching.image_A.keyPoints, matching.image_B.imageRGB, matching.image_B.keyPoints, matching.matches, None, **draw_params)
        
        # cv.imshow("Homog_"+str(matching.image_A.id)+"_"+str(matching.image_B.id), goodMatchesImage)

        return maskInliers.reshape(-1)
            