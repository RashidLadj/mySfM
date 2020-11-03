import cv2 as cv
import numpy as np

from config import *
from Utils import *

class Triangulation:
    
    def __init__(self):
        pass

    def Triangulate(self, matching):
        ###################################################
        ##        Compute 3D points from 2D Images       ##
        ###################################################
        if(configuration["undistort_point"]):
            """ Projection matrix if points are undistort (not need camera matrix) """
            ProjectionMatrix_A = matching.image_A.absoluteTransformation["projection"]
            ProjectionMatrix_B = matching.image_B.absoluteTransformation["projection"]

            # Triangulation
            points4dHomogeneous = cv.triangulatePoints(ProjectionMatrix_A, ProjectionMatrix_B, matching.prec_pts_norm.copy().reshape(-1, 1, 2), matching.curr_pts_norm.copy().reshape(-1, 1, 2))
            points3d = cv.convertPointsFromHomogeneous(points4dHomogeneous.T).reshape(-1,3) 

        else:
            """ Projection matrix if points are not undistort (need camera matrix) """
            ProjectionMatrix_A = matching.image_A.cameraMatrix @ matching.image_A.absoluteTransformation["projection"]
            ProjectionMatrix_B = matching.image_B.cameraMatrix @ matching.image_B.absoluteTransformation["projection"]

            # Triangulation
            points4dHomogeneous = cv.triangulatePoints(ProjectionMatrix_A, ProjectionMatrix_B, matching.prec_pts.copy().reshape(-1, 1, 2), matching.curr_pts.copy().reshape(-1, 1, 2))
            points3d = cv.convertPointsFromHomogeneous(points4dHomogeneous.T).reshape(-1,3)  


        print ("\t\tJust Compute Reprojection Error")
        """ Reprojection Error Image A"""
        reprojection_error_A = compute_reprojection_error_2(matching.image_A.absoluteTransformation["transform"], points3d, matching.prec_pts, matching.image_A.cameraMatrix)
        print ("\t\t\tReprojection Error of Image A after triangulation (with outliers) --> ", reprojection_error_A)
        
        """ Reprojection Error Image B """
        reprojection_error_B = compute_reprojection_error_2(matching.image_B.absoluteTransformation["transform"], points3d, matching.curr_pts, matching.image_B.cameraMatrix)
        print ("\t\t\tReprojection Error of Image B after triangulation (with outliers) --> ", reprojection_error_B)

            
        #''' https://github.com/xdspacelab/openvslam/blob/master/src/openvslam/camera/perspective.cc#L155 '''
        ###################################################
        ## check if reprojected point has positive depth ##
        ## filter object points to have reasonable depth ##
        ###################################################

        MAX_DEPTH = 6.
        index_to_Remove = np.argwhere((points3d[:, 2] < 0) | (points3d[:, 2] > MAX_DEPTH) )

        # Update inliers points
        points3d           = np.delete(points3d              , index_to_Remove, axis=0)
        matching.prec_pts  = np.delete(matching.prec_pts     , index_to_Remove, axis=0)
        matching.curr_pts  = np.delete(matching.curr_pts     , index_to_Remove, axis=0)

        """ Reprojection Error Image A"""
        reprojection_error_A = compute_reprojection_error_2(matching.image_A.absoluteTransformation["transform"], points3d, matching.prec_pts, matching.image_A.cameraMatrix)
        print ("\t\t\tReprojection Error of Image A after triangulation (without outliers) --> ", reprojection_error_A)
        
        """ Reprojection Error Image B """
        reprojection_error_B = compute_reprojection_error_2(matching.image_B.absoluteTransformation["transform"], points3d, matching.curr_pts, matching.image_B.cameraMatrix)
        print ("\t\t\tReprojection Error of Image B after triangulation (without outliers) --> ", reprojection_error_B)

        return points3d