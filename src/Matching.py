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
        


    def __match(self) :
        self.matches, Result = self.matchingConfig.matchMethod(self.image_A, self.image_B)
        self.prec_pts  = np.float32([ self.image_A.keyPoints[m.queryIdx].pt for m in self.matches ]).reshape(-1, 2)
        self.prec_desc = [ self.image_A.des[m.queryIdx] for m in self.matches ]
        self.curr_pts  = np.float32([ self.image_B.keyPoints[m.trainIdx].pt for m in self.matches ]).reshape(-1, 2)
        self.curr_desc = [ self.image_B.des[m.trainIdx] for m in self.matches ] 
        self.__undistortPoint()
       
        # cv.imshow("matches_" + str(image_A.id)+"_"+ str(image_B.id), image_result)
        # cv.waitKey(0)
    

    def __undistortPoint(self):
        ##################################################################
        ##  ''' Undistort for case where K1 != K2  (Online phase) '''   ##
        ##################################################################
        # Undistort points for Essential Matrix calculation --> CameraMatrix will be Identity matrix
        self.prec_pts_norm = cv.undistortPoints((self.prec_pts), cameraMatrix = self.image_A.cameraMatrix, distCoeffs = None)
        self.curr_pts_norm = cv.undistortPoints((self.curr_pts), cameraMatrix = self.image_B.cameraMatrix, distCoeffs = None)
    

    def permute_prec_curr(self):
        self.image_A,        self.image_B        = self.image_B,      self.image_A
        self.prec_pts,       self.curr_pts      = self.curr_pts,      self.prec_pts
        self.prec_pts_norm,  self.curr_pts_norm = self.curr_pts_norm, self.prec_pts_norm
        self.prec_desc,      self.curr_desc,    = self.curr_desc,     self.prec_desc  
        # if you need draw
        def permuteValue(item):
            item.trainIdx, item.queryIdx = item.trainIdx, item.queryIdx
        [permuteValue(item) for item in self.matches]


    def remove_outliers(self, Fundamentalmat, Essentialmat):
        ''' Compute Fundamental matrix '''
        self.fundamentalMatrix_inliers(Fundamentalmat)
        nb_matches_fund = sum(self.inliers_mask)

        ''' Compute Essential matrix '''
        self.essentialMatrix_inliers(Essentialmat)
        print ("\tMatching points beetwin {} and {} : Nb Matches (No optimize) == {} , Nb Matches_Fund == {}, Nb matches_Essen == {}".format(self.image_A.id, self.image_B.id, len(self.matches), nb_matches_fund, sum(self.inliers_mask)))
    
    def fundamentalMatrix_inliers(self, Fundamentalmat):
        if configuration["enable_fundamentalMat"]:
            self.inliers_mask = Fundamentalmat.compute_FundamentalMatrix(self.curr_pts, self.prec_pts, self.image_B, self.image_A, self.matches)  
            assert not (sum(self.inliers_mask) == 0), "not enough points to calculate F"
            # print ("\tCompute fundamental matrix: {} inliers / {} ".format(sum(self.inliers_mask), len(self.matches)))
            self.update_inliers_mask()


    def homographyMatrix_inliers(self, Homographymat):
        Homog_mask = Homographymat.compute_HomographyMatrix(self.curr_pts, self.prec_pts, self.image_B, self.image_A, self.matches)  
        self.homog_mask_len = np.sum(Homog_mask)
        print ("\tMatching points beetwin {} and {} : Nb Matches (No optimize) == {} , Nb Matches (Homography) == {} ".format(self.image_A.id, self.image_B.id, len(self.matches), self.homog_mask_len))


    def essentialMatrix_inliers(self, Essentialmat):
        self.inliers_mask, self.essential_matrix, self.focal_avg = Essentialmat.compute_EssentialMatrix(self.curr_pts, self.prec_pts, self.curr_pts_norm, self.prec_pts_norm, self.image_B, self.image_A)
        assert not (sum(self.inliers_mask) == 0), "not enough points to calculate E"
        # print ("\tCompute essential matrix: {} inliers / {} ".format(sum(self.inliers_mask), len(self.matches)))
        
    def update_inliers_mask(self):
        # Update inliers points and descriptors for triangulation
        self.prec_pts       = self.prec_pts[self.inliers_mask.ravel() > 0]
        self.prec_pts_norm  = self.prec_pts_norm[self.inliers_mask.ravel() > 0]
        
        self.curr_pts       = self.curr_pts[self.inliers_mask.ravel() > 0]
        self.curr_pts_norm  = self.curr_pts_norm[self.inliers_mask.ravel() > 0]
        
        self.inliers_mask   = np.ones((len(self.curr_pts),), dtype=int)


    # /***********************************************************/
    # /*****  Retrieve exesiting 3D points in current image  *****/          
    # /***********************************************************/
    def retrieve_existing_points(self):
        ###########################################################
        ##              ''' Intersection 3D 2D '''               ##
        ##                ''' Filter Matching '''                ##
        ###########################################################
        print("\tcompute intersection between ", len(self.image_A.points_2D_used), " and", len(self.prec_pts), "points for projection")
        _, index_A, self.index_toRemove_forTriangulate = intersect2D(np.asarray(self.image_A.points_2D_used), np.asarray(self.prec_pts)) 
        if (len(index_A) < 7): 
            print ("\t Pas assez de points !!!")
            return    ## to Verify
        """ Retrieve points for solver (inter_3d_pts & inter_curr_pts) """
        inter_3d_pts       = np.asarray(self.image_A.points_3D_used) [index_A, :].reshape(-1, 3)
        inter_curr_pts     = np.asarray(self.curr_pts)               [self.index_toRemove_forTriangulate, :].reshape(-1, 2) 
        inter_curr_pts_norm= np.asarray(self.curr_pts_norm)          [self.index_toRemove_forTriangulate, :].reshape(-1, 2)
        print("\t\tnumber of 3D points to project on the new image for estimate points is ", len(inter_3d_pts), "points")

        return inter_3d_pts, inter_curr_pts, inter_curr_pts_norm

        
    def points_for_triangulate(self):
        print ("\t\tPrepare points for triangulate (not in intersection)")
        """ Retrieve points for triangulate (not in intersection) """
        self.prec_pts      = np.delete(self.prec_pts      , self.index_toRemove_forTriangulate, axis=0).reshape(-1, 2)
        self.prec_pts_norm = np.delete(self.prec_pts_norm , self.index_toRemove_forTriangulate, axis=0).reshape(-1, 2)
        self.curr_pts      = np.delete(self.curr_pts      , self.index_toRemove_forTriangulate, axis=0).reshape(-1, 2)
        self.curr_pts_norm = np.delete(self.curr_pts_norm , self.index_toRemove_forTriangulate, axis=0).reshape(-1, 2)
        print("\t\tpoint to triangulate ", len(self.prec_pts)) 

        self.inliers_mask = np.ones((len(self.curr_pts),), dtype=int)


    def generate_landmarks(self):
        print ("\tGenerate 3D_points using Triangulation")
        self.update_inliers_mask()   
        # # Take just inliers for triangulation
        p_cloud  = Triangulation().Triangulate(self)

        self.image_A.points_2D_used  = np.append(self.image_A.points_2D_used, self.prec_pts  , axis = 0)
        self.image_A.points_3D_used  = np.append(self.image_A.points_3D_used, p_cloud        , axis = 0)
    
        self.image_B.points_2D_used  = np.append(self.image_B.points_2D_used, self.curr_pts  , axis = 0)
        self.image_B.points_3D_used  = np.append(self.image_B.points_3D_used, p_cloud        , axis = 0)
        
        print("\tnumber of 3D-point image_A is ", len(self.image_A.points_2D_used), "and image B is ", len(self.image_B.points_2D_used))

