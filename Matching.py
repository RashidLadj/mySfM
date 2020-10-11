import cv2 as cv
import numpy as np
from PoseEstimation import *
from Triangulation import *


class Matching:
    def __init__(self, MatchingConfig, image_A, image_B):
        self.matchingConfig = MatchingConfig        
        self.image_A, self.image_B= image_A, image_B      
        self.__match()
        self.inliers_mask = np.ones(len(self.matches))


    def __match(self) :
        self.matches, Result = self.matchingConfig.matchMethod(self.image_A, self.image_B)
        self.src_pts  = np.around(np.float32([ self.image_A.keyPoints[m.queryIdx].pt for m in self.matches ]).reshape(-1, 2))
        self.src_desc = [ self.image_A.des[m.queryIdx] for m in self.matches ]
        self.dst_pts  = np.around(np.float32([ self.image_B.keyPoints[m.trainIdx].pt for m in self.matches ]).reshape(-1, 2))
        self.dst_desc = [ self.image_B.des[m.trainIdx] for m in self.matches ] 
        self.__undistortPoint()
        print ("    Matching points beetwin", self.image_A.id, "and",  self.image_B.id, "--> Number of Matches (No optimize) ==", len(self.matches))
        # cv.imshow("matches_" + str(image_A.id)+"_"+ str(image_B.id), image_result)
        # cv.waitKey(0)
    

    ##################################################################
    ##  ''' Undistort for case where K1 != K2  (Online phase) '''   ##
    ##################################################################
    def __undistortPoint(self):
        # Undistort points for Essential Matrix calculation --> CameraMatrix will be Identity matrix
        self.src_pts_norm = cv.undistortPoints(np.float32(self.src_pts), cameraMatrix = self.image_A.cameraMatrix, distCoeffs = None)
        self.dst_pts_norm = cv.undistortPoints(np.float32(self.dst_pts), cameraMatrix = self.image_B.cameraMatrix, distCoeffs = None)
        # affine = self.CameraMatrix[0:2, :]
        # self.src_ptsNorm = cv.transform(self.src_ptsNorm, affine)
        # self.dst_ptsNorm = cv.transform(self.dst_ptsNorm, affine)
    

    def computePose_2D2D(self, Essentialmat, initialize_step = False):
        ''' Compute Essential matrix '''
        self.inliers_mask = Essentialmat.compute_EssentialMatrix(self)
        assert not (sum(self.inliers_mask) == 0), "not enough points to calculate E"
        print ("\tCompute essential matrix: ", sum(self.inliers_mask),"inliers /",len(self.matches))

        ''' Estimate Pose '''
        inliers_count, Rotation_Mat, Transl_Vec, self.inliers_mask = PoseEstimation().RecoverPose_from_2D2D(self, Essentialmat)
        print ("\tRecover-pose 2D-2D: ", inliers_count,"inliers /",len(self.src_pts))

        ''' set absolute Pose '''
        if (initialize_step):
            self.image_A.setAbsolutePose(None, np.eye(3), np.zeros(3))
            self.image_B.setAbsolutePose(self.image_A, Rotation_Mat, Transl_Vec)


        ###########################################################
        ##    ''' Generate 3D_points using Triangulation '''     ##
        ###########################################################
        print ("\tGenerate 3D_points using Triangulation")
        ''' Remove all outliers and triangulate'''
        self.__update_inliers_mask()   # Take just inliers for triangulation
        points3d, index_to_Remove = Triangulation().Triangulate(self)
        
        # Update inliers points and descriptors
        p_cloud                = np.delete(points3d                 , index_to_Remove, axis=0)

        src_pts_inliers_F      = np.delete(self.src_pts     , index_to_Remove, axis=0)
        src_pts_norm_        = np.delete(self.src_pts_norm, index_to_Remove, axis=0)
        src_desc_inliers_F     = np.delete(self.src_desc    , index_to_Remove, axis=0)

        dst_pts_inliers_F      = np.delete(self.dst_pts     , index_to_Remove, axis=0)
        dst_pts_norm_F       = np.delete(self.dst_pts_norm, index_to_Remove, axis=0)
        dst_desc_inliers_F     = np.delete(self.dst_desc    , index_to_Remove, axis=0)

    
        print ("\tnumber of points 3D between", self.image_A.id, "and",  self.image_B.id, "after triangulation is", len(p_cloud),"points")

        if (initialize_step):
            self.image_A.points_2D_used  = src_pts_inliers_F
            self.image_A.descriptor_used = src_desc_inliers_F
            self.image_A.points_3D_used  = p_cloud

            self.image_B.points_2D_used  = dst_pts_inliers_F
            self.image_B.descriptor_used = dst_desc_inliers_F
            self.image_B.points_3D_used  = p_cloud

    
    def __update_inliers_mask(self):
        # Update inliers points and descriptors for triangulation
        self.src_pts       = self.src_pts[self.inliers_mask.ravel() > 0]
        self.src_pts_norm  = self.src_pts_norm[self.inliers_mask.ravel() > 0]
        self.src_desc      = [ desc  for desc, i in zip (self.src_desc, np.arange(len(self.src_desc))) if self.inliers_mask[i] > 0]
        self.dst_pts       = self.dst_pts[self.inliers_mask.ravel() > 0]
        self.dst_pts_norm  = self.dst_pts_norm[self.inliers_mask.ravel() > 0]
        self.dst_desc      = [ desc  for desc, i in zip (self.dst_desc, np.arange(len(self.dst_desc))) if self.inliers_mask[i] > 0]


""" Just Configuration Class """
class MatchingConfig:
    def __init__(self, Matcher, Symmetric, Symmetric_Type, CrossCheck, Lowes_ratio):
        self.matcher = Matcher                  # FlannMatcher or BFmatcher
        self.symmetric = Symmetric              # True or False
        self.symmetric_Type = Symmetric_Type    # Intersection or union
        self.crosscheck = CrossCheck            # true or false
        self.lowes_ratio = Lowes_ratio          # Lowes_ratio for knnMatch
        self.matchMethod = self.__config_method()   # method to use [MatchingSimple, MatchingIntersection , MatchingUnion]      


    def __config_method(self):
        if not self.symmetric:
            return self.__MatchingSimple
        elif self.symmetric and self.symmetric_Type == "intersection" :
            return self.__MatchingIntersection
        elif self.symmetric and self.symmetric_Type == "union" :
            return self.__MatchingUnion
        else:
            print ("Configuration False")
            return None


    ###################################################################
    ##       ''' Retrieve matches from A to B (Assymetric) '''       ##
    ###################################################################
    def __MatchingSimple(self, image_A, image_B):
        if self.crosscheck:
            # Methode One
            matches = self.matcher.match(image_A.des, image_B.des)
            # take just 2/3 ?? (A revoir)
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw All matches.
            image_matches = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        else:
            # Methode Two
            matches = self.matcher.knnMatch(image_A.des, image_B.des, k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < self.lowes_ratio * n.distance:
                    good.append([m])
                    #matches.append(m)
            matches = [item[0] for item in good]

            # Draw All matches.
            image_matches = cv.drawMatchesKnn(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,good, None, flags=2)
        
        # Sort them in the order of their distance.
        # matches = sorted(matches, key = lambda x:x.distance)

        return matches, image_matches    


    def __permuteValue(self, item):
        temp = item.trainIdx
        item.trainIdx = item.queryIdx
        item.queryIdx = temp


    def __areEqual(self, itemAB, itemBA):
        return itemAB.trainIdx == itemBA.trainIdx and itemAB.queryIdx == itemBA.queryIdx and itemAB.distance == itemBA.distance and itemAB.imgIdx == itemBA.imgIdx


    ####################################################################
    ##   ''' Retrieve matches Symmetricly and take intersection '''   ##
    ####################################################################
    def __MatchingIntersection(self, image_A, image_B):
        if self.crosscheck:
            # Methode One
            ''' Match Img A to img B '''
            matches_AB = self.matcher.match(image_A.des, image_B.des)
            
            ''' Match Img B to img A '''
            matches_BA = self.matcher.match(image_B.des, image_A.des)

        else:
            # Methode Two
            ''' Match Img A to img B '''
            matches_AB = self.matcher.knnMatch(image_A.des, image_B.des, k=2)
            
            # Apply ratio test
            good_AB = []
            for m, n in matches_AB:
                if m.distance < self.lowes_ratio * n.distance:
                    good_AB.append([m])  # If need to use drawKnnMatch
                    #matches_ABF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_AB = [(item[0]) for item in good_AB]

            ''' Match Img B to img A '''
            matches_BA = self.matcher.knnMatch(image_B.des, image_A.des, k=2)

            # Apply ratio test
            good_BA = []
            for m, n in matches_BA:
                if m.distance < self.lowes_ratio * n.distance:
                    good_BA.append([m])  # If need to use drawKnnMatch
                    #matches_BAF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_BA = [(item[0]) for item in good_BA]

        matches_BA_ConvertToAB = matches_BA.copy()
        [self.__permuteValue(item) for item in matches_BA_ConvertToAB]

        intersectionMatches = []
        for itemAB in matches_AB:
            for itemBA in matches_BA_ConvertToAB:
                if self.__areEqual(itemAB, itemBA) :   # Je n'ai pas trouvé un autre moyen de faire la comparaison
                    intersectionMatches.append(itemAB)

        # Sort them in the order of their distance.
        intersectionMatches = sorted(intersectionMatches, key = lambda x: x.distance)   

        # take just 2/3 ?? (A revoir)
        # matches = sorted(matches, key = lambda x:x.distance)

        goodInter = intersectionMatches.copy()
        goodInterKnn = [[item] for item in intersectionMatches]

        # Matching 
        good_Without_Optimizer = len(intersectionMatches)
        image_matches = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodInter, None, flags=2) if self.crosscheck else cv.drawMatchesKnn(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodInterKnn, None, flags=2)

        return intersectionMatches, image_matches    


    ####################################################################
    ##       ''' Retrieve matches Symmetricly and take union '''      ##
    ####################################################################
    def __MatchingUnion(self, image_A, image_B):
        if self.crosscheck:
                # Methode One
            ''' Match Img A to img B '''
            matches_AB = self.matcher.match(image_A.des, image_B.des)
            
            ''' Match Img B to img A '''
            matches_BA = self.matcher.match(image_B.des, image_A.des)

        else:
            # Methode Two
            ''' Match Img A to img B '''
            matches_AB = self.matcher.knnMatch(image_A.des, image_B.des, k=2)

            # Apply ratio test
            good_AB = []
            for m,n in matches_AB:
                if m.distance < self.lowes_ratio * n.distance:
                    good_AB.append([m])  # If need to use drawKnnMatch
                    #matches_ABF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_AB = [(item[0]) for item in good_AB]

            ''' Match Img B to img A '''
            matches_BA = self.matcher.knnMatch(image_B.des, image_A.des, k=2)
            # Apply ratio test
            good_BA = []
            for m,n in matches_BA:
                if m.distance < self.lowes_ratio * n.distance:
                    good_BA.append([m])  # If need to use drawKnnMatch
                    #matches_BAF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_BA = [(item[0]) for item in good_BA]

        matches_BA_ConvertToAB = matches_BA.copy()
        [self.__permuteValue(item) for item in matches_BA_ConvertToAB]

        unionMatches = []
        [unionMatches.append(item) for item in matches_AB]
        for itemBA in matches_BA_ConvertToAB:
            exist = False
        for item in unionMatches:
            if self.__areEqual(item, itemBA) :
                exist = True
            break   # Je n'ai pas trouvé un autre moyen de faire la comparaison
        if not exist:
            unionMatches.append(itemBA)

        # Sort them in the order of their distance.
        unionMatches = sorted(unionMatches, key = lambda x: x.distance)

        goodUnion = unionMatches.copy()
        goodUnionKnn = [[item] for item in unionMatches]

        # Matching 
        good_Without_Optimizer = len(unionMatches)
        image_matches = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodUnion, None, flags=2) if self.crosscheck else cv.drawMatchesKnn(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodUnionKnn, None, flags=2)

        return unionMatches, image_matches 

      
    

 


    