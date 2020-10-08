import cv2 as cv
import numpy as np


class EssentialMatrix:
    def __init__(self, methodOptimizer = cv.RANSAC, threshold = 3.0, CameraMatrixArray = np.identity(3)):
        self.methodOptimizer = methodOptimizer
        self.threshold = threshold
        self.CameraMatrixArray = CameraMatrixArray


    #####################################################################
    ## NOTE: Pose is calculated from dst to src so it's the same for E ##
    #####################################################################
    def compute_EssentialMatrix (self, src_pts, dst_pts):
        NB_Matching_Threshold = 5
        if len(src_pts) >= NB_Matching_Threshold:

            ''' Methode 1 -->  Without normalize  points [-1, 1] '''
            self.EssentialMat, maskInliers = cv.findEssentialMat(dst_pts, src_pts, self.CameraMatrixArray, method = self.methodOptimizer, prob = 0.999, threshold = self.threshold) 

            # ''' Methode 2 -->  Without normalize points [-1, 1] --> A eviter si focalX != focalY  ''' 
            # principalPoint = (self.CameraMatrixArray[0][2], self.CameraMatrixArray[1][2])
            # focalx = self.CameraMatrixArray[0][0]
            # EssentialMat, maskInliers = cv.findEssentialMat(self.dst_pts, self.dst_pts, focal = focalx, pp = principalPoint, method = self.methodOptimizer, prob = 0.999, threshold = self.threshold)

            # ''' Methode 3 -->  UndistordPoints and normalize points [-1, 1]  '''
            # # Bizarrre .... 
            # self.NormalizePoint()
            # EssentialMat, maskInliers = cv.findEssentialMat(self.dst_ptsNorm, self.dst_ptsNorm, focal = 1., pp = (0., 0.), method = self.methodOptimizer, prob = 0.999, threshold = self.threshold)
        
        else:
            print("Not enough matches are found - %d < %d" , (len(self.matches),NB_Matching_Threshold))
            return None, None

        return maskInliers


    def RecoverPose_3D_Points(self, src_pts, dst_pts):
        #####################################################
        ## Compute rotation and translation of camera 2    ##
        ## Generate transformation matrix using R&t matrix ## 
        ##      ''' Camera Matrix is mondatory '''         ##
        #####################################################
            points, Rot_RP, Transl_RP, mask_RP = cv.recoverPose(self.EssentialMat, dst_pts, src_pts, self.CameraMatrixArray)
            # print ("\nCompute Transformation With recoverPose : ( ", points, " point inliers )")
            # print ("    \nRotation \n", R) 
            # print ("    \ntranslation \n", t) 

            # transform = np.vstack((np.hstack((Rot_RP, transl_RP)), [0, 0, 0, 1]))
            # print ("    \nTransformation Matrix \n", transform) 
            return points, Rot_RP, Transl_RP, mask_RP
    
    
    def Triangulate(self, image_A, image_B, src_pts, dst_pts):
        ###################################################
        ##        Compute 3D points from 2D Images       ##
        ###################################################
            """ Projection matrix  """
            ProjectionMatrix_1 = self.CameraMatrixArray @ image_A.absoluteTransformation["projection"]
            ProjectionMatrix_2 = self.CameraMatrixArray @ image_B.absoluteTransformation["projection"]

            # Triangulation
            points4dHomogeneous = cv.triangulatePoints(ProjectionMatrix_1, ProjectionMatrix_2, src_pts.copy().reshape(-1, 1, 2), dst_pts.copy().reshape(-1, 1, 2))
            points3d = cv.convertPointsFromHomogeneous(points4dHomogeneous.T).reshape(-1,3)  


        #''' https://github.com/xdspacelab/openvslam/blob/master/src/openvslam/camera/perspective.cc#L155 '''
        ###################################################
        ## check if reprojected point has positive depth ##
        ## filter object points to have reasonable depth ##
        ###################################################
            MAX_DEPTH = 4.
            index_to_Remove = np.argwhere((points3d[:, 2] < 0) | (points3d[:, 2] > MAX_DEPTH) )
            
            return points3d, index_to_Remove
