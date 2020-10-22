import cv2 as cv
import numpy as np

from PoseEstimation import *
from Triangulation import *
from Utils import *
from config import *

class Matching:
    def __init__(self, MatchingConfig, image_A, image_B):
        self.matchingConfig = MatchingConfig        
        self.image_A, self.image_B= image_A, image_B      
        self.__match()
        self.inliers_mask = np.ones(len(self.matches))


    def __match(self) :
        self.matches, Result = self.matchingConfig.matchMethod(self.image_A, self.image_B)
        self.prec_pts  = np.float32([ self.image_A.keyPoints[m.queryIdx].pt for m in self.matches ]).reshape(-1, 2)
        self.prec_desc = [ self.image_A.des[m.queryIdx] for m in self.matches ]
        self.curr_pts  = np.float32([ self.image_B.keyPoints[m.trainIdx].pt for m in self.matches ]).reshape(-1, 2)
        self.curr_desc = [ self.image_B.des[m.trainIdx] for m in self.matches ] 
        self.__undistortPoint()
       
        # cv.imshow("matches_" + str(image_A.id)+"_"+ str(image_B.id), image_result)
        # cv.waitKey(0)
    

    ##################################################################
    ##  ''' Undistort for case where K1 != K2  (Online phase) '''   ##
    ##################################################################
    def __undistortPoint(self):
        # Undistort points for Essential Matrix calculation --> CameraMatrix will be Identity matrix
        self.prec_pts_norm = cv.undistortPoints((self.prec_pts), cameraMatrix = self.image_A.cameraMatrix, distCoeffs = None)
        self.curr_pts_norm = cv.undistortPoints((self.curr_pts), cameraMatrix = self.image_B.cameraMatrix, distCoeffs = None)
        # affine = self.CameraMatrix[0:2, :]
        # self.prec_ptsNorm = cv.transform(self.prec_ptsNorm, affine)
        # self.curr_ptsNorm = cv.transform(self.curr_ptsNorm, affine)
    

    def permute_prec_curr(self):
        self.image_A,        self.image_B        = self.image_B,      self.image_A
        self.prec_pts,       self.curr_pts      = self.curr_pts,      self.prec_pts
        self.prec_pts_norm,  self.curr_pts_norm = self.curr_pts_norm, self.prec_pts_norm
        self.prec_desc,      self.curr_desc,    = self.curr_desc,     self.prec_desc  

    def setHomog_mask(self, val):
        self.homog_mask_len = val
        print ("    Matching points beetwin", self.image_A.id, "and",  self.image_B.id, "--> Number of Matches (No optimize) ==", len(self.matches),  "--> Number of Matches (Homography) ==", val)


    def computePose_2D2D(self, Fundamentalmat, Essentialmat, initialize_step = False):
        if configuration["enable_fundamentalMat"]:
            ''' Compute Fundamental matrix '''
            self.inliers_mask = Fundamentalmat.compute_FundamentalMatrix(self)
            assert not (sum(self.inliers_mask) == 0), "not enough points to calculate F"
            print ("\tCompute fundamental matrix: ", sum(self.inliers_mask),"inliers /",len(self.matches))

        self.__update_inliers_mask()
        self.inliers_mask = np.ones(len(self.prec_pts))

        ''' Compute Essential matrix '''
        self.inliers_mask, self.focal_avg = Essentialmat.compute_EssentialMatrix(self)
        assert not (sum(self.inliers_mask) == 0), "not enough points to calculate E"
        print ("\tCompute essential matrix: ", sum(self.inliers_mask),"inliers /",len(self.prec_pts))

        ''' Estimate Pose '''
        inliers_count, _, _, self.inliers_mask = PoseEstimation().EstimatePose_from_2D2D(self, Essentialmat)
        print ("\tEstimate-pose 2D-2D: ", inliers_count,"inliers /",len(self.matches))
        if (initialize_step):
            self.image_A.setAbsolutePose(None, np.eye(3), np.zeros(3))


        ###########################################################
        ##    ''' Generate 3D_points using Triangulation '''     ##
        ###########################################################
        print ("\tGenerate 3D_points using Triangulation")
        ''' Remove all outliers and triangulate'''
        self.__update_inliers_mask()   # Take just inliers for triangulation
        points3d, index_to_Remove  = Triangulation().Triangulate(self)
        
        # Update inliers points and descriptors
        p_cloud            = np.delete(points3d         , index_to_Remove, axis=0)

        prec_pts_inliers_F  = np.delete(self.prec_pts     , index_to_Remove, axis=0)
        prec_pts_norm_      = np.delete(self.prec_pts_norm, index_to_Remove, axis=0)
        prec_desc_inliers_F = np.delete(self.prec_desc    , index_to_Remove, axis=0)

        curr_pts_inliers_F  = np.delete(self.curr_pts     , index_to_Remove, axis=0)
        curr_pts_norm_F     = np.delete(self.curr_pts_norm, index_to_Remove, axis=0)
        curr_desc_inliers_F = np.delete(self.curr_desc    , index_to_Remove, axis=0)


        

    
        print ("\tnumber of points 3D between", self.image_A.id, "and",  self.image_B.id, "after triangulation is", len(p_cloud),"points")

        if (initialize_step):
            self.image_A.points_2D_used  = prec_pts_inliers_F
            self.image_A.descriptor_used = prec_desc_inliers_F
            self.image_A.points_3D_used  = p_cloud

            self.image_B.points_2D_used  = curr_pts_inliers_F
            self.image_B.descriptor_used = curr_desc_inliers_F
            self.image_B.points_3D_used  = p_cloud

    
    def computePose_3D2D(self, Fundamentalmat, Essentialmat):
        if configuration["enable_fundamentalMat"]:
            ''' Compute Fundamental matrix '''
            self.inliers_mask = Fundamentalmat.compute_FundamentalMatrix(self)
            assert not (sum(self.inliers_mask) == 0), "not enough points to calculate F"
            print ("\tCompute fundamental matrix: ", sum(self.inliers_mask),"inliers /",len(self.matches))

        self.__update_inliers_mask()
        self.inliers_mask = np.ones(len(self.prec_pts))

        ''' Compute Essential matrix '''
        self.inliers_mask, self.focal_avg = Essentialmat.compute_EssentialMatrix(self)
        assert not (sum(self.inliers_mask) == 0), "not enough points to calculate E"
        print ("\tCompute essential matrix: ", sum(self.inliers_mask),"inliers /",len(self.matches))

        self.__update_inliers_mask()   # Take just inliers for triangulation

        ###########################################################
        ##              ''' Intersection 3D 2D '''               ##
        ##                ''' Filter Matching '''                ##
        ###########################################################
        print("\tcompute intersection between ", len(self.image_A.points_2D_used), "and", len(self.prec_pts), "points for projection")
        _, index_A, index_B = intersect2D(np.asarray(self.image_A.points_2D_used), np.asarray(self.prec_pts)) 
        if (len(index_A) < 4):
            return False
        """ Retrieve points for solver (inter_pts_3D_PnP & curr_pts_PnP) """
        self.inter_pts_3D_PnP  = np.asarray(self.image_A.points_3D_used)[index_A, :].reshape(-1, 3)
        self.curr_pts_B_2D_PnP  = np.asarray(self.curr_pts)               [index_B, :].reshape(-1, 2) 
        self.curr_pts_B_norm_PnP= np.asarray(self.curr_pts_norm)          [index_B, :].reshape(-1, 2)
        # pt_B_desc_Current  = np.asarray(curr_pts_inliers_E)[index_B, :]  ## MANQUE
        print("\t\tnumber of 3D points to project on the new image for estimate points is ", len(self.inter_pts_3D_PnP), "points")

        ''' Estimate Pose '''
        inliers_count, _, _, self.inliers_mask = PoseEstimation().EstimatePose_from_3D2D(self, Essentialmat)
        print ("\tEstimate-pose 3D-2D")

        ''' Update Existing 3D point to add for current image '''
        self.inter_pts_3D_PnP  = np.asarray(self.inter_pts_3D_PnP)  [self.inliers_mask, :].reshape(-1, 3)
        self.curr_pts_B_2D_PnP  = np.asarray(self.curr_pts_B_2D_PnP)  [self.inliers_mask, :].reshape(-1, 2) 
        self.curr_pts_B_norm_PnP= np.asarray(self.curr_pts_B_norm_PnP)[self.inliers_mask, :].reshape(-1, 2)

        ###########################################################
        ##    ''' Generate 3D_points using Triangulation '''     ##
        ###########################################################
        print ("\tGenerate 3D_points using Triangulation")
        """ Retrieve points for triangulate (not in intersection) """
       
        self.prec_pts      = np.delete(self.prec_pts      , index_B, axis=0).reshape(-1, 2)
        self.prec_pts_norm = np.delete(self.prec_pts_norm , index_B, axis=0).reshape(-1, 2)
        self.curr_pts      = np.delete(self.curr_pts      , index_B, axis=0).reshape(-1, 2)
        self.curr_pts_norm = np.delete(self.curr_pts_norm , index_B, axis=0).reshape(-1, 2)
        print("\t\tpoint to triangulate ", len(self.prec_pts)) 
        points3d, index_to_Remove = Triangulation().Triangulate(self)
        
        # Update inliers points and descriptors
        p_cloud            = np.delete(points3d         , index_to_Remove, axis=0)

        prec_pts_inliers_F  = np.delete(self.prec_pts     , index_to_Remove, axis=0)
        prec_pts_norm_      = np.delete(self.prec_pts_norm, index_to_Remove, axis=0)
        prec_desc_inliers_F = np.delete(self.prec_desc    , index_to_Remove, axis=0)

        curr_pts_inliers_F  = np.delete(self.curr_pts     , index_to_Remove, axis=0)
        curr_pts_norm_F     = np.delete(self.curr_pts_norm, index_to_Remove, axis=0)
        curr_desc_inliers_F = np.delete(self.curr_desc    , index_to_Remove, axis=0)

       
        print ("\tnumber of 3D-points between", self.image_A.id, "and",  self.image_B.id, "after triangulation is", len(p_cloud),"points")
        
        self.image_A.points_2D_used  = np.append(self.image_A.points_2D_used, prec_pts_inliers_F, axis = 0)
        self.image_A.points_3D_used  = np.append(self.image_A.points_3D_used, p_cloud, axis = 0)
        self.image_B.points_2D_used  = self.curr_pts_B_2D_PnP.copy()
        self.image_B.points_2D_used  = np.append(self.image_B.points_2D_used, curr_pts_inliers_F, axis = 0)

        self.image_B.points_3D_used  = self.inter_pts_3D_PnP
        self.image_B.points_3D_used  = np.append(self.image_B.points_3D_used, p_cloud, axis = 0)
        
        print("\tnumber of 3D-point image_A is ", len(self.image_A.points_2D_used), "and image B is ", len(self.image_B.points_2D_used))

        return True

    
    def __update_inliers_mask(self):
        # Update inliers points and descriptors for triangulation
        self.prec_pts       = self.prec_pts[self.inliers_mask.ravel() > 0]
        self.prec_pts_norm  = self.prec_pts_norm[self.inliers_mask.ravel() > 0]
        
        self.curr_pts       = self.curr_pts[self.inliers_mask.ravel() > 0]
        self.curr_pts_norm  = self.curr_pts_norm[self.inliers_mask.ravel() > 0]
        


    def computePose_2D2D_Scale(self, Essentialmat):
        ''' Compute Essential matrix '''
        self.inliers_mask, _ = Essentialmat.compute_EssentialMatrix(self)
        assert not (sum(self.inliers_mask) == 0), "not enough points to calculate E"
        print ("\tCompute essential matrix: ", sum(self.inliers_mask),"inliers /",len(self.matches))

        ''' Estimate Pose '''
        inliers_count, Rotation_Mat, Transl_Vec, self.inliers_mask = PoseEstimation().EstimatePose_from_2D2D_scale(self, Essentialmat)
        print ("\tEstimate-pose 2D-2D: ", inliers_count,"inliers /",len(self.prec_pts))
        

        ###########################################################
        ##    ''' Generate 3D_points using Triangulation '''     ##
        ###########################################################
        print ("\tGenerate 3D_points using Triangulation")
        ''' Remove all outliers and triangulate'''
        self.__update_inliers_mask()   # Take just inliers for triangulation
        points3d, index_to_Remove = Triangulation().Triangulate(self)
        
        # Update inliers points and descriptors
        p_cloud            = np.delete(points3d         , index_to_Remove, axis=0)

        prec_pts_inliers_F  = np.delete(self.prec_pts     , index_to_Remove, axis=0)
        prec_pts_norm_      = np.delete(self.prec_pts_norm, index_to_Remove, axis=0)
        # prec_desc_inliers_F = np.delete(self.prec_desc    , index_to_Remove, axis=0)

        curr_pts_inliers_F  = np.delete(self.curr_pts     , index_to_Remove, axis=0)
        curr_pts_norm_F     = np.delete(self.curr_pts_norm, index_to_Remove, axis=0)
        # curr_desc_inliers_F = np.delete(self.curr_desc    , index_to_Remove, axis=0)

        
        _, index_last, index_new = intersect2D(self.image_A.points_2D_used, prec_pts_inliers_F)
        
        p_cloud_new = p_cloud[index_new]
        p_cloud_old = self.image_A.points_3D_used[index_last]
        
        """ Thibaud Method """
        scale_ratio_last = [np.linalg.norm(p_cloud_old[i] - p_cloud_old[j]) for i in range(0, len(p_cloud_old) - 1) for j in range(i+1, len(p_cloud_old))]
        scale_ratio_new  = [np.linalg.norm(p_cloud_new [i] - p_cloud_new [j]) for i in range(0, len(p_cloud_new)  - 1) for j in range(i+1, len(p_cloud_new ))]
        scale = [(x / y) for x, y in zip(scale_ratio_last , scale_ratio_new) ]
        print (np.array(scale).reshape(-1))

        """ StackOverflow Method """
        # min_x_new, max_x_new, min_pt_x_new, max_pt_x_new = cv.minMaxLoc(p_cloud_new[:, 0])
        # min_y_new, max_y_new, min_pt_y_new, max_pt_y_new = cv.minMaxLoc(p_cloud_new[:, 1])
        # min_z_new, max_z_new, min_pt_z_new, max_pt_z_new = cv.minMaxLoc(p_cloud_new[:, 2])


        # min_x_old, max_x_old, min_pt_x_old, max_pt_x_old = cv.minMaxLoc(p_cloud_old[:, 0])
        # min_y_old, max_y_old, min_pt_y_old, max_pt_y_old = cv.minMaxLoc(p_cloud_old[:, 1])
        # min_z_old, max_z_old, min_pt_z_old, max_pt_z_old = cv.minMaxLoc(p_cloud_old[:, 2])


        # print ("__-__", min_x_new, min_x_old, min_y_new, p_cloud_old[:, 1][62], min_z_new, min_z_old)
        # print ("__-__", max_x_new, max_x_old, max_y_new, max_y_old, p_cloud_old[:, 2][8], max_z_old)

        # print ("__-__", min_pt_x_new, min_pt_x_old, min_pt_y_new, min_pt_y_old, min_pt_z_new, min_pt_z_old)
        # print ("__-__", max_pt_x_new, max_pt_x_old, max_pt_y_new, max_pt_y_old, max_pt_z_new, max_pt_z_old)


        # min_y_old = p_cloud_old[:, 1][62]
        # max_z_new = p_cloud_old[:, 2][8]

        # xScale = (max_x_old - min_x_old) / (max_x_new - min_x_new) 
        # yScale = (max_y_old - min_y_old) / (max_y_new - min_y_new) 
        # zScale = (max_z_old - min_z_old) / (max_z_new - min_z_new) 

        # print ("__-__", xScale, yScale, zScale )

        print (Transl_Vec)
        print (np.mean(scale))
        Transl_Vec = Transl_Vec * np.mean(scale)#([xScale, yScale, zScale]).reshape(-1, 1)  
        print (Transl_Vec)
        ''' set absolute Pose '''
        self.image_B.setAbsolutePose(self.image_A, Rotation_Mat, Transl_Vec)
        self.image_B.absoluteTransformation["transform"] = self.image_A.absoluteTransformation["transform"] @ self.image_B.absoluteTransformation["transform"]
        self.image_B.absoluteTransformation["projection"] = projection_from_transformation(self.image_B.absoluteTransformation["transform"])

        print ("\tnumber of points 3D between", self.image_A.id, "and",  self.image_B.id, "after triangulation is", len(p_cloud),"points")

        points3d, index_to_Remove = Triangulation().Triangulate(self)
         # Update inliers points and descriptors
        p_cloud            = np.delete(points3d         , index_to_Remove, axis=0)

        prec_pts_inliers_F  = np.delete(self.prec_pts     , index_to_Remove, axis=0)
        prec_pts_norm_      = np.delete(self.prec_pts_norm, index_to_Remove, axis=0)
        # prec_desc_inliers_F = np.delete(self.prec_desc    , index_to_Remove, axis=0)

        curr_pts_inliers_F  = np.delete(self.curr_pts     , index_to_Remove, axis=0)
        curr_pts_norm_F     = np.delete(self.curr_pts_norm, index_to_Remove, axis=0)
        # curr_desc_inliers_F = np.delete(self.curr_desc    , index_to_Remove, axis=0)


        self.image_A.points_2D_used  = np.append(self.image_A.points_2D_used, prec_pts_inliers_F, axis = 0)
        self.image_A.points_3D_used  = np.append(self.image_A.points_3D_used, p_cloud, axis = 0)

        self.image_B.points_2D_used  = curr_pts_inliers_F
        # self.image_B.points_2D_used  = np.append(self.image_B.points_2D_used, curr_pts_inliers_F, axis = 0)
        self.image_B.points_3D_used  = p_cloud
        # self.image_B.points_3D_used  = np.append(self.image_B.points_3D_used, p_cloud, axis = 0)
