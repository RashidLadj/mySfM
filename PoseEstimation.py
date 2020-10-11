import cv2 as cv
import numpy as np

from config import *


class PoseEstimation:

    def __init__(self):
        pass
        

    #############################################################
    ##      ''' PoseEstimation 2D-2D - Initialization'''       ##
    #############################################################
    def RecoverPose_from_2D2D(self, matching, EssentialMatrix):
        #####################################################
        ## Compute rotation and translation of camera 2    ##
        ##      ''' Camera Matrix is mondatory '''         ##
        ## points undistort camera matrix == identity matrix##
        #####################################################
        if(configuration["undistort_point"]):
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.dst_pts_norm, matching.src_pts_norm, cameraMatrix = np.eye(3), mask = matching.inliers_mask)
        else:
            inliers_count, Rotatio_nMat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.dst_pts, matching.src_pts, cameraMatrix = matching.image_A.cameraMatrix, mask = matching.inliers_mask)

        return inliers_count, Rotation_Mat, Transl_Vec, mask_inliers
