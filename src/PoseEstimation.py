import cv2 as cv
import numpy as np

from config import *
from Utils import *


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
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.curr_pts_norm.reshape(-1, 2), matching.prec_pts_norm.reshape(-1, 2), cameraMatrix = np.eye(3), mask = matching.inliers_mask)
        else:
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.curr_pts, matching.prec_pts, cameraMatrix = matching.image_A.cameraMatrix, mask = matching.inliers_mask)

        assert CheckCoherentRotation(Rotation_Mat), "\tRotation Erreur in EstimatePose_from_2D2D" 
        
        ''' set absolute Pose '''
        matching.image_B.setAbsolutePose(matching.image_A, Rotation_Mat, Transl_Vec)

        return inliers_count, Rotation_Mat, Transl_Vec, mask_inliers

    
    #############################################################
    ##      ''' PoseEstimation 2D-2D - Initialization'''       ##
    #############################################################
    def EstimatePose_from_2D2D_scale(self, matching, EssentialMatrix):
        #####################################################
        ## Compute rotation and translation of camera 2    ##
        ##      ''' Camera Matrix is mondatory '''         ##
        ## points undistort camera matrix == identity matrix##
        #####################################################
        if(configuration["undistort_point"]):
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.curr_pts_norm.reshape(-1, 2), matching.prec_pts_norm.reshape(-1, 2), cameraMatrix = np.eye(3), mask = matching.inliers_mask)
        else:
            inliers_count, Rotation_Mat, Transl_Vec, mask_inliers = cv.recoverPose(EssentialMatrix.EssentialMat, matching.curr_pts, matching.prec_pts, cameraMatrix = matching.image_A.cameraMatrix, mask = matching.inliers_mask)
        
        
        # matching.image_B.setRelativPose(matching.image_A, Rotation_Mat, Transl_Vec)
        # matching.image_B.relativeTransformation["transform"] = matching.image_A.absoluteTransformation["transform"] @ matching.image_B.absoluteTransformation["transform"]
        # # matching.image_B.relativeTransformation["projection"] = projection_from_transformation(matching.image_B.absoluteTransformation["transform"])

        assert CheckCoherentRotation(Rotation_Mat), "\tRotation Erreur in EstimatePose_from_2D2D_scale" 
        ''' set absolute Pose '''
        matching.image_B.setAbsolutePose(matching.image_A, Rotation_Mat, Transl_Vec)
        matching.image_B.absoluteTransformation["transform"] = matching.image_A.absoluteTransformation["transform"] @ matching.image_B.absoluteTransformation["transform"]
        matching.image_B.absoluteTransformation["projection"] = projection_from_transformation(matching.image_B.absoluteTransformation["transform"])

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
            curr_pts = matching.curr_pts_B_norm_PnP.copy().reshape(-1, 1, 2)
            CameraMatrix = np.eye(3)
            reprojectionError = 3./matching.focal_avg
        else:
            curr_pts = matching.curr_pts_B_2D_PnP.copy().reshape(-1, 1, 2)
            CameraMatrix = matching.image_A.cameraMatrix
            reprojectionError = 3.

        pts_3D = matching.inter_pts_3D_PnP.copy().reshape(-1, 1, 3)

        ###########################################################
        ##      ''' Retrieve Transformation of Image_B '''       ##
        ###########################################################
        sizeInput = len(pts_3D)
        inliers_index = []

        if (not configuration["pnpsolver_method"]):
            _ , absolute_rVec, absolute_tVec = cv.solvePnP(pts_3D, curr_pts, CameraMatrix, None, flags = cv.SOLVEPNP_ITERATIVE)  # All points, no inliers
            
        else:
            _, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(pts_3D, curr_pts, CameraMatrix, None, iterationsCount = 200000, reprojectionError = reprojectionError, flags = cv.SOLVEPNP_EPNP)

        """ Remove outliers for calculate reprojection error """
        pts_3D = np.asarray(pts_3D)[inliers_index, :].reshape(-1, 3)
        curr_pts  = np.asarray(curr_pts)[inliers_index, :].reshape(-1, 2)

        
        project_points, _  = cv.projectPoints(pts_3D.T, absolute_rVec, absolute_tVec, CameraMatrix, distCoeffs=None, )
        reprojection_error = np.linalg.norm(curr_pts - project_points.reshape(-1, 2)) / len(project_points)
        print ("\t\t\tReprojection Error After Solve PnP ransac is ", reprojection_error, "with ", len(inliers_index), "inliers / ", sizeInput)

        """ Retrieve Absolute Transformation of current image """
        absolute_Rt, _      = cv.Rodrigues(absolute_rVec)
        absolute_Rotation   = absolute_Rt.T

        assert CheckCoherentRotation(absolute_Rotation), "\tRotation Erreur in EstimatePose_from_3D2D" 

        absolute_translation    = - absolute_Rotation @ absolute_tVec
        matching.image_B.setAbsolutePose(matching.image_A, absolute_Rotation, absolute_translation)


        """ Reprojection Error """
        reprojection_error_A = compute_reprojection_error(matching.image_B.absoluteTransformation["transform"], pts_3D, curr_pts, matching.image_B.cameraMatrix)
        print ("\t\t\tReprojection Error After Solve PnP ransac is ", reprojection_error_A, "with ", len(inliers_index), "inliers / ", sizeInput)
        

        

       
        
        """ Create Mask """
        mask_inliers = np.zeros(sizeInput, dtype = np.int32)
        mask_inliers[inliers_index.reshape(-1)] = 1

        return len(inliers_index), absolute_Rotation, absolute_translation, mask_inliers


