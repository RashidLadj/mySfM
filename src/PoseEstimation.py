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
    def FindPoseEstimation_RecoverPose(self, essential_matrix, src_pts, dst_pts, src_pts_norm, dst_pts_norm, cameraMatrix_A, cameraMatrix_B, inliers_mask = None):
        # /*******************************************************/
        # /**   Compute rotation and translation of new image   **/
        # /**         ''' Camera Matrix is mondatory '''        **/
        # /** points undistort camera matrix == identity matrix **/
        # /**            Use undistort is recommended           **/
        # /*******************************************************/

        if(configuration["undistort_point"] or not np.array_equal(cameraMatrix_A, cameraMatrix_B)):
            inliers_count, Rotation_Mat, Transl_Vec, _ = cv.recoverPose(essential_matrix, src_pts_norm, dst_pts_norm, cameraMatrix = np.eye(3),      mask = inliers_mask)
        else:
            inliers_count, Rotation_Mat, Transl_Vec, _ = cv.recoverPose(essential_matrix, src_pts,      dst_pts,      cameraMatrix = cameraMatrix_A, mask = inliers_mask)

        if not CheckCoherentRotation(Rotation_Mat):
            print ("\tRotation Erreur in EstimatePose_from_2D2D")
            return
        
        return inliers_count, Rotation_Mat, Transl_Vec

    
    #############################################################
    ##       ''' PoseEstimation 2D-2D - Incremental '''        ##
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
        
        assert CheckCoherentRotation(Rotation_Mat), "\tRotation Erreur in EstimatePose_from_2D2D_scale" 
        ''' set absolute Pose '''
        matching.image_B.setAbsolutePose(matching.image_A, Rotation_Mat, Transl_Vec)
        matching.image_B.absoluteTransformation["transform"] = matching.image_A.absoluteTransformation["transform"] @ matching.image_B.absoluteTransformation["transform"]
        matching.image_B.absoluteTransformation["projection"] = projection_from_transformation(matching.image_B.absoluteTransformation["transform"])

        return inliers_count, Rotation_Mat, Transl_Vec, mask_inliers


    #############################################################
    ##       ''' PoseEstimation 3D-2D - Incremental '''        ##
    #############################################################
    def FindPoseEstimation_pnp(self, ppcloud, imgPoints, imgPoints_norm, cameraMatrix):
        #####################################################
        ##  Compute rotation and translation of new image  ##
        ##      ''' Camera Matrix is mondatory '''         ##
        ## points undistort camera matrix == identity matrix##
        #####################################################

        if(len(ppcloud) <= 7 or len(imgPoints) <= 7 or len(ppcloud) != len(imgPoints)) : 
            print("couldn't find [enough] corresponding cloud points... (only ", len(ppcloud), ")")
            return None
        
        if(configuration["undistort_point"]):
            imgPoints_ = imgPoints_norm.copy().reshape(-1, 1, 2)
            CameraMatrix = np.eye(3)
            reprojectionError = 3./cameraMatrix[0][0]
        else:
            imgPoints_ = imgPoints.copy().reshape(-1, 1, 2)
            CameraMatrix = cameraMatrix
            reprojectionError = 3.

        ppcloud_       = ppcloud.copy().reshape(-1, 1, 3)
        imgPoints_Proj = imgPoints.copy()

        ###########################################################
        ##      ''' Retrieve Transformation of Image_B '''       ##
        ###########################################################
        sizeInput = len(ppcloud_)

        # inliers_index = []
        # if (not configuration["pnpsolver_method"]):
        #     _ , absolute_rVec, absolute_tVec = cv.solvePnP(ppcloud_, imgPoints_, CameraMatrix, None, flags = cv.SOLVEPNP_ITERATIVE)  # All points, no inliers
        # else:
        done, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(ppcloud_, imgPoints_, CameraMatrix, None, iterationsCount = 20000, reprojectionError = reprojectionError, flags = cv.SOLVEPNP_EPNP)
        if not done: return

        ## if not outliers, solvepnpransac retourn none ##
        if inliers_index is None:
            """ Not outliers --> create inliers index """
            inliers_index = np.array([i for i in range (len(ppcloud_))])

        reprojection_error = compute_reprojection_error_1(absolute_rVec, absolute_tVec, ppcloud_, imgPoints_Proj, cameraMatrix)
        print ("\t\t\tReprojection Error After Solve PnP ransac (With outlier) is ", reprojection_error, "with ", len(inliers_index), "inliers / ", sizeInput)

        """ Remove outliers for calculate reprojection error """
        ppcloud_        = np.asarray(ppcloud_)       [inliers_index, :].reshape(-1, 3)
        imgPoints_      = np.asarray(imgPoints_)     [inliers_index, :].reshape(-1, 2)
        imgPoints_Proj  = np.asarray(imgPoints_Proj) [inliers_index, :].reshape(-1, 2)
        
        reprojection_error = compute_reprojection_error_1(absolute_rVec, absolute_tVec, ppcloud_, imgPoints_Proj, cameraMatrix)
        print ("\t\t\tReprojection Error After Solve PnP ransac (Without outliers) is ", reprojection_error, "with ", len(inliers_index), "inliers / ", sizeInput)

        # if(len(inliers)==0): #get inliers
        #     for i in range (lenprojected3D)) : 
        #         if norm(projected3D[i]-imgPoints[i]) < 10.0 : 
        #             inliers.push_back(i);

        if len(inliers_index) < (len(imgPoints)/5.0):    # if len(inliers) < 20% of all
            print("not enough inliers to consider a good pose (", len(inliers_index), "/", len(imgPoints), ")")
            return
        
        if(cv.norm(absolute_tVec) > 200.0):
            # this is bad...
            print("estimated camera movement is too big, skip this camera")
            return

        """ Retrieve Absolute Transformation of current image """
        absolute_Rt, _       = cv.Rodrigues(absolute_rVec)
        absolute_Rotation    = absolute_Rt.T
        absolute_translation = - absolute_Rotation @ absolute_tVec

        if not CheckCoherentRotation(absolute_Rotation):
            print("\trotation is incoherent. we should try a different base view ...2" )
            return

        """ Create Mask """
        mask_inliers = np.zeros(sizeInput, dtype = np.int32)
        mask_inliers[inliers_index.reshape(-1)] = 1

        return len(inliers_index), absolute_Rotation, absolute_translation, mask_inliers