import cv2 as cv
import numpy as np

from config import *


#########################################################################################################
## Source:                                                                                             ##
## - https://www.programcreek.com/python/?code=zju3dv%2FGIFT%2FGIFT-master%2Ftrain%2Fevaluation.py#    ##
##   For 2 camera                                                                                      ##
#########################################################################################################


class EssentialMatrix:
    def __init__(self, methodOptimizer = cv.RANSAC, threshold = 3.0):
        self.methodOptimizer = methodOptimizer
        self.threshold = threshold 
        ## If Two Cameras are used : -> Undistore points --> k= Identity ##


    #####################################################################
    ## NOTE: Pose is calculated from dst to src so it's the same for E ##
    ## If points aren't undistort --> Use K = cameraMatrix                 ##
    ## If points are undistort    --> Use K = Identity Matrix              ##
    #####################################################################
    def compute_EssentialMatrix (self, matching):
        focal_avg = (matching.image_A.cameraMatrix[0][0] + matching.image_B.cameraMatrix[0][0]) / 2
        
        NB_Matching_Threshold = 5
        if len(matching.matches) >= NB_Matching_Threshold:
            if (np.array_equal(matching.image_A.cameraMatrix, matching.image_B.cameraMatrix)): 
                ## Same Result ##
                if(configuration["undistort_point"]):
                    self.EssentialMat, self.maskInliers = cv.findEssentialMat(matching.curr_pts_norm, matching.prec_pts_norm, np.eye(3), method = self.methodOptimizer, prob = 0.999, threshold = self.threshold / focal_avg, mask = matching.inliers_mask) 
                else:    
                    self.EssentialMat, self.maskInliers = cv.findEssentialMat(matching.curr_pts, matching.prec_pts, matching.image_A.cameraMatrix, method = self.methodOptimizer, prob = 0.999, threshold = self.threshold, mask = matching.inliers_mask) 

            else:
                ## Same Result ##
                if cv.__version__ >= '4.5.0':
                    ## New Method will be integrate in next  release of OpenCV##
                    """https://docs.opencv.org/master/d9/d0c/group__calib3d.html#gafafd52c0372b12dd582597bfb1330430"""
                    self.EssentialMat, self.maskInliers = cv.findEssentialMat(matching.curr_pts, matching.prec_pts, cameraMatrix1 = matching.image_B.cameraMatrix, cameraMatrix2 = matching.image_A.cameraMatrix, method = self.methodOptimizer, prob = 0.999, threshold = self.threshold, mask = matching.inliers_mask) 

                else:
                    self.EssentialMat, self.maskInliers = cv.findEssentialMat(matching.curr_pts_norm, matching.prec_pts_norm, focal = 1., pp = (0., 0.), method = self.methodOptimizer, prob = 0.999, threshold = self.threshold/focal_avg, mask = matching.inliers_mask)
                    self.EssentialMat, self.maskInliers = cv.findEssentialMat(matching.curr_pts_norm, matching.prec_pts_norm, np.eye(3), method = self.methodOptimizer, prob = 0.999, threshold = self.threshold/focal_avg, mask = matching.inliers_mask) 
                    
        else:
            print("Not enough matches are found - %d < %d" , (len(self.matches),NB_Matching_Threshold))
            return np.zeros(len(matching.prec_pts)), focal_avg

        return self.maskInliers.reshape(-1), focal_avg
    
    

