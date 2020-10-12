import cv2 as cv
import numpy as np

from config import *


class PoseEstimation:

    def __init__(self):
        pass
        

    #############################################################
    ##      ''' PoseEstimation 2D-2D - Initialization'''       ##
    #############################################################
    def EstimatePose_from_2D2D(self, matching, EssentialMatrix):
        #####################################################
        ## Compute rotation and translation of camera 2    ##
        ##      ''' Camera Matrix is mondatory '''         ##
        ## points undistort camera matrix == identity matrix##
        #####################################################
        if(configuration["undistort_point"]):
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.dst_pts_norm.reshape(-1, 2), matching.src_pts_norm.reshape(-1, 2), cameraMatrix = np.eye(3), mask = matching.inliers_mask)
        else:
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.dst_pts, matching.src_pts, cameraMatrix = matching.image_A.cameraMatrix, mask = matching.inliers_mask)

        ''' set absolute Pose '''
        matching.image_B.setAbsolutePose(matching.image_A, Rotation_Mat, Transl_Vec)

        return inliers_count, Rotation_Mat, Transl_Vec, mask_inliers

    
    #############################################################
    ##       ''' PoseEstimation 3D-2D - Incremental '''        ##
    #############################################################
    def EstimatePose_from_3D2D(self, matching, EssentialMatrix):
        #####################################################
        ## Compute rotation and translation of camera 2    ##
        ##      ''' Camera Matrix is mondatory '''         ##
        ## points undistort camera matrix == identity matrix##
        #####################################################
        if(configuration["undistort_point"]):
            dst_pts = matching.dst_pts_B_norm_PnP.copy().reshape(-1, 1, 2)
            CameraMatrix = np.eye(3)
        else:
            dst_pts = matching.dst_pts_B_PnP.copy().reshape(-1, 1, 2)
            CameraMatrix = matching.image_A.cameraMatrix
        pts_3D = matching.inter_pts_3D_PnP.copy().reshape(-1, 1, 3)

        #     inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.dst_pts_norm, matching.src_pts_norm, cameraMatrix = np.eye(3), mask = matching.inliers_mask)
        # else:
        #     inliers_count, Rotatio_nMat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.dst_pts, matching.src_pts, cameraMatrix = matching.image_A.cameraMatrix, mask = matching.inliers_mask)

        ###########################################################
        ##      ''' Retrieve Transformation of Image_B '''       ##
        ###########################################################
        sizeInput = len(pts_3D)
        inliers_index = []
        if (not configuration["pnpsolver_method"]):
            _ , absolute_rVec, absolute_tVec = cv.solvePnP(pts_3D, dst_pts, CameraMatrix, None, flags = cv.SOLVEPNP_ITERATIVE)  # All points, no inliers
            
        else:
            _, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(pts_3D, dst_pts, CameraMatrix, None, iterationsCount = 200000, reprojectionError = 3./matching.focal_avg, flags = cv.SOLVEPNP_EPNP)

        """ Remove outliers for calculate reprojection error """
        pts_3D = np.asarray(pts_3D)[inliers_index, :].reshape(-1, 3)
        dst_pts  = np.asarray(dst_pts)[inliers_index, :].reshape(-1, 2)

        """ Reprojection Error """
        project_points, _ = cv.projectPoints(pts_3D.T, absolute_rVec, absolute_tVec, CameraMatrix, distCoeffs=None, )
        reprojection_error = np.linalg.norm(dst_pts - project_points.reshape(-1, 2)) / len(project_points)
        print ("\tJust Compute Reprojection Error")
        print ("\tReprojection Error After Solve PnP ransac ( not default params ) --> ", reprojection_error, "with ", len(inliers_index), "inliers / ", sizeInput)

        """ Retrieve Absolute Transformation of current image """
        absolute_Rt, _ = cv.Rodrigues(absolute_rVec)
        absolute_Rotation    = absolute_Rt.T

        # proj_matrix = np.hstack((absolute_Rt, absolute_tVec))
        absolute_translation    = - absolute_Rotation @ absolute_tVec
        matching.image_B.setAbsolutePose(matching.image_A, absolute_Rotation, absolute_translation)
        
        """ Create Mask """
        mask_inliers = np.eye(sizeInput, dtype = np.int32)
        mask_inliers[inliers_index] = 1

        return len(inliers_index), absolute_Rotation, absolute_translation, mask_inliers


