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
            EssentialMat, maskInliers = cv.findEssentialMat(dst_pts, src_pts, self.CameraMatrixArray, method = self.methodOptimizer, prob = 0.999, threshold = self.threshold) 


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

        return EssentialMat, maskInliers
        
    
    # def decomposeEssentialMatrix_3D_Points(self):
    #     R1, R2, t = cv.decomposeEssentialMat(self.E)

    #     print ("\nCompute Transformation With decomposeEssentialMat : ")
    #     print ("    \nRotation_1 \n", R1) 
    #     print ("    \nRotation_2 \n", R2) 
    #     print ("    \ntranslation \n", t) 

    #     # Suppose relative pose between two cameras with generate projection matrix. 
    #     # Fix the first camera to (0, 0, 0)
    #     P1 = np.concatenate((self.CameraMatrixArray, np.zeros((3,1))), axis=1) #   K [I | 0]
    #     Trans_pos = [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]
    #     P2s =  [np.dot(self.CameraMatrixArray, np.concatenate(Trans_pos[0], axis=1)), # K[R1 |  t]
    #             np.dot(self.CameraMatrixArray, np.concatenate(Trans_pos[1], axis=1)), # K[R1 | -t]
    #             np.dot(self.CameraMatrixArray, np.concatenate(Trans_pos[2], axis=1)), # K[R2 |  t]
    #             np.dot(self.CameraMatrixArray, np.concatenate(Trans_pos[3], axis=1))] # K[R2 | -t]

    #     obj_pts_per_cam = []

    #     # calculate the 3D points with each projection
    #     for P2, Tr in zip(P2s, Trans_pos):
    #         obj_pt = cv.triangulatePoints(P1, P2, self.src_pts_inliers.copy().reshape(-1, 1, 2), self.dst_pts_inliers.copy().reshape(-1, 1, 2))
    #         obj_pt /= obj_pt[3]

    #         # check if reprojected point has positive depth
    #         obj_pts = []
    #         for obj in obj_pt.T:
    #             if obj[2] > 0:
    #                 obj_pts.append([obj[0], obj[1], obj[2]])
                
    #         obj_pts_per_cam.append(obj_pts)

    #         # filter object points to have reasonable depth
    #         # MAX_DEPTH = 6.
    #         # pts_3D = []
    #         # for pt in obj_pts:
    #         #     if pt[2] < MAX_DEPTH:
    #         #         pts_3D.append(pt)

    #         # obj_pts_per_cam.append(pts_3D)
           
    #     # take the projection that gives us the most 3D points
    #     cam_idx = np.array([len(obj_pts_per_cam[0]),len(obj_pts_per_cam[1]),len(obj_pts_per_cam[2]),len(obj_pts_per_cam[3])])
    #     print ("    Projecion ", [len(obj_pts_per_cam[0]),len(obj_pts_per_cam[1]),len(obj_pts_per_cam[2]),len(obj_pts_per_cam[3])])
    #     # best_cam_idx = np.argmax(cam_idx)
    #     best_cam_idx = cam_idx.argsort()[-1:][::-1][0]
    #     # best_cam_idx = 3

    #     print ("    Best Projection is ", (best_cam_idx)) 
    #     max_pts = len(obj_pts_per_cam[best_cam_idx])
        
    #     print ("    Best Projection is ", (best_cam_idx), " with ", max_pts, " points " )

    #     # filter object points to have reasonable depth
    #     MAX_DEPTH = 4.
    #     pts_3D = []
    #     for pt in obj_pts_per_cam[best_cam_idx]:
    #         if pt[2] < MAX_DEPTH:
    #             pts_3D.append(pt)

    #     RotMatrix, TranslVector = Trans_pos[best_cam_idx]
    #     P2 = P2s[best_cam_idx]
    #     print ("    Projection \n", P2)

    #     #return RotMatrix, TranslVector, 3D_Points, ProjectionMatrix2
    #     return RotMatrix, TranslVector, pts_3D, P2


    def RecoverPose_3D_Points(self, EssentialMat, src_pts, dst_pts):
        #####################################################
        ## Compute rotation and translation of camera 2    ##
        ## Generate transformation matrix using R&t matrix ## 
        ##      ''' Camera Matrix is mondatory '''         ##
        #####################################################
            points, Rot_RP, Transl_RP, mask_RP = cv.recoverPose(EssentialMat, dst_pts, src_pts, self.CameraMatrixArray)
            # print ("\nCompute Transformation With recoverPose : ( ", points, " point inliers )")
            # print ("    \nRotation \n", R) 
            # print ("    \ntranslation \n", t) 

            # transform = np.vstack((np.hstack((Rot_RP, transl_RP)), [0, 0, 0, 1]))
            # print ("    \nTransformation Matrix \n", transform) 
            return points, Rot_RP, Transl_RP, mask_RP
    
    def Triangulate(self, Rot, Transl, src_pts, dst_pts):
        ###################################################
        ##        Compute 3D points from 2D Images       ##
        ###################################################
            # Projection matrix
            P1 = np.concatenate((self.CameraMatrixArray, np.zeros((3,1))), axis=1) #   K [I | 0]
            P2 = np.dot(self.CameraMatrixArray, np.concatenate((Rot.T, (-Rot.T @ Transl)), axis=1))
            #print ("    \nProjection Matrix of image 2 \n", P2) 
            
            # Triangulation
            points4dHomogeneous = cv.triangulatePoints(P1, P2, src_pts.copy().reshape(-1, 1, 2), dst_pts.copy().reshape(-1, 1, 2))
            points3d = cv.convertPointsFromHomogeneous(points4dHomogeneous.T).reshape(-1,3)  

        ###################################################
        ## check if reprojected point has positive depth ##
        ###################################################
            # Retrieve index of points whose depth is negative
            index_Negatif_Depth = np.argwhere(points3d[:, 2] < 0)

        ###################################################
        ## filter object points to have reasonable depth ##
        ###################################################
            MAX_DEPTH = 4.
            # Retrieve index of points whose depth exceeds 4.
            index_Max_Depth = np.argwhere(points3d[:, 2] > MAX_DEPTH)

            index_to_Remove = np.concatenate((index_Negatif_Depth, index_Max_Depth))
            # points3d = np.delete(points3d , index_to_Remove, axis=0).tolist()
            return points3d, index_to_Remove

    #     #####################################################
    #     ##          Calculate Reprojection Error           ##
    #     ##        Necessite transformation inverse         ##
    #     ##     transformation_inverse  = |R.T  -R.T*t|     ##
    #     ##                               |0      1   |     ##
    #     #####################################################
    #         ''' https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/ '''
    #         # # Reproject back into the two cameras
    #         rvec1, _ = cv.Rodrigues(np.eye(3))
    #         rvec2, _ = cv.Rodrigues(R.T)

    #         print ("rvec2 --> ", rvec2)

    #         tvec2 = np.dot(-R.T,t)
            

    #         p1, _ = cv.projectPoints(np.asarray(points3d_F).T, rvec1, np.zeros((3,1)), self.CameraMatrixArray, distCoeffs=None)
    #         p2, _ = cv.projectPoints(np.asarray(points3d_F).T, rvec2, np.dot(-R.T,t) , self.CameraMatrixArray, distCoeffs=None)
            
    #         # for p_1, k_1, p_2, k_2 in zip (p1, self.src_pts_inliers, p2, self.dst_pts_inliers):
    #         #     print ("p1 == ", p_1, k_1, p_2, k_2)

    #         # measure difference between original image point and reporjected image point 
    #     # Using numpy normalize
    #         reprojection_error1 = np.linalg.norm(self.src_pts_inliers.reshape(-1, 2) - p1.reshape(-1, 2)) / len(p1)
    #         reprojection_error2 = np.linalg.norm(self.dst_pts_inliers.reshape(-1, 2) - p2.reshape(-1, 2)) / len(p2)
    #     # Using openCV normalize
    #         # reprojection_error1 = cv.norm(self.src_pts_inliers.reshape(-1, 2) , p1.reshape(-1, 2), cv.NORM_L2) / len(p1)
    #         # reprojection_error2 = cv.norm(self.dst_pts_inliers.reshape(-1, 2) , p2.reshape(-1, 2), cv.NORM_L2) / len(p2)
    #         print("    \nReprojection Error Image_A --> ", reprojection_error1)
    #         print("    Reprojection Error Image_B --> ", reprojection_error2, "\n")

    #     #####################################################
    #     ##                    SOLVE PnP                    ##
    #     #####################################################
    #         _, rVec, tVec = cv.solvePnP(points3d_F, self.dst_pts_inliers, self.CameraMatrixArray, None)
    #         print ("rvec2 --> \n", rVec, "\n", rvec2)
    #         p1, _ = cv.projectPoints(np.asarray(points3d_F).T, rVec, tVec, self.CameraMatrixArray, distCoeffs=None)
    #         reprojection_error1 = np.linalg.norm(self.src_pts_inliers.reshape(-1, 2) - p1.reshape(-1, 2)) / len(p1)
    #         print("    \nReprojection Error Image_A --> ", reprojection_error1)
            

    #         # print ("rVec , tVec --> ",rVec, tVec)
    #         Rt, _ = cv.Rodrigues(rVec)
    #         R_ = np.asarray(Rt.T)
            
    #         # /***** R_ == R *****/         # [O][1] pas pareil  :/
    #         # /***** tVec == tvec2 *****/   # Egaux
    #         # /***** R_ == R *****/         #
    #         # /***** R_ == R *****/         #
    #         print ("youpi --> \n", R_,"\n", R)
    #         print ("tvecAncient - nouveau  --> ", tvec2, tVec)
    #         print ("tvecAncient - nouveau  --> ", -np.dot(np.linalg.inv(R.T), tvec2), +t)
    #         pos = -np.dot(np.linalg.inv(Rt) , np.asarray(tVec))
    #         print ("youpi --> \n", pos,"\n", t)

    #         # # print ("youpi -->", points3d_F)


    #         ############################################################################################
    #         ## _, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)      ##
    #         ## Rt = cv2.Rodrigues(rvec)                                                               ##
    #         ## R = Rt.transpose()                                                                     ##
    #         ## pos = -R * tVec                                                                        ##
    #         ############################################################################################
    #         t = pos
    #         R = R_
    #         return R, t, points3d_F, P2


    # # def decomposeSVD(self):
    # #     ''' planar '''
    # #     # https://docs.opencv.org/master/d0/d92/samples_2cpp_2tutorial_code_2features2D_2Homography_2pose_from_homography_8cpp-example.html#a24Ò
    # #     # Normalization to ensure that ||c1|| = 1
    # #     W, U, Vt = cv.SVDecomp(self.E)

    # #     singular_values_ratio = np.abs(W[0] / W[1])
    # #     if singular_values_ratio > 1.0 : 
    # #         singular_values_ratio = 1.0 / singular_values_ratio # flip ratio to keep it [0,1]
    # #     if singular_values_ratio < 0.7 :
    # #         print("singular values are too far apart\n")


    # #     #https://github.com/libmv/libmv/blob/8040c0f6fa8e03547fd4fbfdfaf6d8ffd5d1988b/src/libmv/multiview/fundamental.cc#L302-L338
    # #     # Last column of U is undetermined since d = (a a 0).
    # #     if cv.determinant(U) < 0 :
    # #         #U[:, 2] *= -1
    # #         print("khra1")
    # #         U *= -1
    # #     # Last row of Vt is undetermined since d = (a a 0).
    # #     if cv.determinant(Vt) < 0 :
    # #         #Vt[2, :] *= -1
    # #         print("khra2")
    # #         Vt *= -1
    # #     #std::cout << "vt:\n" << vt << std::endl;

    # #     w = np.zeros((3, 3))
    # #     w[0][1] = -1
    # #     w[1][0] = 1
    # #     w[2][2] = 1

    # #     R1 = U * w * Vt; #HZ 9.19
    # #     R2 = U * w.T * Vt; 
    # #     t = U[:, 2]; #u3

    # #     if np.abs(cv.determinant(R1)) - 1.0 > 1e-07 :    
    # #         print("det(R) != +-1.0, this is not a rotation matrix" )

    # #     print ("\nCompute Transformation With SVD : ")
    # #     print ("    Rotatiot_1 \n", R1) 
    # #     print ("    Rotatiot_2 \n", R2) 
    # #     print ("    translation_ \n", t.reshape(-1, 1)) 

    # #     return R1, R2, t.reshape(-1, 1)


    # def generate_3D_points(self, method):
    #     ################################################################################################
    #     ## Retrieve transformation of image 2 compared to image 1, as well as the generated 3D points ##
    #     ################################################################################################
    #     if (method == 1):
    #         ''' transform with Decompose Essential Matrix'''
    #         # cette methode propose 4 cas possibles R1_t, R1_t.T, R2_t, R2_t.T donc le meilleur sera choisis en utilisant RecoverPose()
    #         Rot, Trans, points3d, P2 = self.decomposeEssentialMatrix_3D_Points()
            

    #     if (method == 0):
    #         ''' transform with RecoverPose'''
    #         # ==> recoverPose utilise decomposeEssentialMatrix et fait un test sur les combinaisons suivantes R1_t, R1_t.T, R2_t, R2_t.T, et nous retourne le meilleur resultat **/ #
    #         Rot, Trans, points3d, P2 = self.RecoverPose_3D_Points()
        
    #     transform = np.vstack((np.hstack((Rot, Trans)), [0, 0, 0, 1]))

    #     ##################################################################
    #     ## TODO : Save 3D point in each Image with Coordiante 2D and 3D ##
    #     ##################################################################

    #     # self.Image1.add_3Dpoints(points3d,self.src_pts_inliers)
    #     # self.Image2.add_3Dpoints(points3d,self.dst_pts_inliers)

    #     return transform, points3d

    # def computeFundamentalMatrix(self):
    #     # F = inv(K2.T) * self.E * inv(K1)
    #     F_mat = np.dot (np.dot(np.linalg.inv(self.CameraMatrixArray).T, self.E) , np.linalg.inv(self.CameraMatrixArray))
    #     self.getFMatrix = FundamentalMatEstimation.FundamentalMatrixOptimize(self.Image1, self.Image2, matches = self.matches, F = F_mat)
        
    #     self.getFMatrix.src_pts = np.int32([ self.Image1.kp[m.queryIdx].pt for m in self.matches ]).reshape(-1, 2)
    #     self.getFMatrix.dst_pts = np.int32([ self.Image2.kp[m.trainIdx].pt for m in self.matches ]).reshape(-1, 2)




