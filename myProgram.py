###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################
from os import listdir
import os.path
from os import path
from numpy import loadtxt
import open3d as o3d

import json
import itertools

from config import *
from Utils import *

from Image import *
from DescriptorMatcherConfig import *
from Matching import *
from FundamentalMatEstimation import *
from EssentialMatEstimation import *
from PoseEstimation import *



###########################################################
##          ''' Define Detector and Matcher '''          ##
##             ''' Matching configuration '''            ##
##      ''' Config params of Fundamental Matrix '''      ##
##       ''' Config params of Essential Matrix '''       ##
###########################################################
detector, matcher= keyPointDetectorAndMatchingParams(keyPointDetectorMethod = configuration["feature_type"], matcherMethod = configuration["matcher_type"], crosscheck = configuration["matcher_crosscheck"], Max_keyPoint = configuration["max_keyPoint"])      
assert not detector == None and not matcher == None, "Problems !!!"
print ("\nDetector == ", type(detector), " Matcher = ", type(matcher))

print ("\nMatching configuration")
matchingConfig = MatchingConfig(matcher, configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])
# print ("    ", configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])

# print ("\Fundamental Matrix Configuration\n")
# Fundamentalmat = FundamentalMatrix(methodOptimizer = configuration["fundamental_method"])

print ("\nEssential Matrix Configuration")
Essentialmat = EssentialMatrix(methodOptimizer = configuration["essential_method"])


###########################################################
##        ''' Load Camera Matrix and update it '''       ##
##          ''' Load distortion coefficients '''         ##
##                Samsung S7 (1920 x 1080)               ##
###########################################################
''' toutes les images ont la même résolution et sont prises par mon samsung S7 '''
CameraMatrix = loadtxt('CameraMatrixArray.csv', delimiter=',')  #FullHD

''' toutes les images ont la même résolution et sont prises par mon samsung S7 '''
DistortionCoef = None  # not used




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''          PHASE One: prepare Data           '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

ImageFolder = "Images/"
assert path.exists(ImageFolder), 'veuillez verifier le chemin du Folder'
images_name = sorted([f for f in listdir(ImageFolder)])


###########################################################
##                  ''' Create Images '''                ##
###########################################################
print ("Read images")
Images = []
for image in images_name:
    Images.append(Image(ImageFolder, image, configuration["feature_process_size"], CameraMatrix.copy()))
print ("    Images = ", images_name)


###########################################################
##       ''' Calculate KeyPoint and Descriptor '''       ##
###########################################################
print ("\nDetect and compute descriptors")
for image in Images:
    image.keyPointDetector(detector)
    print ('  size of descriptors of', image.id, image.des.shape)
    

###########################################################
##  ''' Compute Matching beetwin each pair of image '''  ##
###########################################################
print ("\nMatching points beetwin each pair of image")

image_Pair = list(itertools.combinations(Images, 2))

matches_vector = []
for image_A, image_B in image_Pair:
    matching_AB = Matching(matchingConfig, image_A, image_B)
    matches_vector.append(matching_AB)
    


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''       PHASE Two: Initialize SfM Data       '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###############################################################
##  ''' Retrieve best candidate pair for initialization '''  ##
###############################################################
initialize_step = True
index_candidate_pair = np.argmax([len (x.matches) for x in matches_vector])
index_candidate_pair = 0                            # temporary value 
matching_AB = matches_vector[index_candidate_pair]
matches_vector.pop(index_candidate_pair)
print ("\nRetrieve best candidate pair for initialization: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")


################################################################
##           ''' Compute Transformation Matrix '''            ##
##      ''' Triangulation and Initialize point-Cloud '''      ##
################################################################
print ("Compute transformation and initialize point-Cloud")
matching_AB.computePose_2D2D(Essentialmat, initialize_step)
initialize_step == False

POINT_DRAW = matching_AB.image_A.points_3D_used




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''      PHASE Three: Incremental SfM Data     '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
              
# ###############################################################
# ## ''' Retrieve best candidate pair for incremetal phase ''' ##
# ###############################################################
# print ("\nRetrieve best candidate pair for incremetal phase")

# index_candidate_pair = np.argmax([len (x[2]) for x in matches_vector])
# image_A, image_B, matches_AB = matches_vector[index_candidate_pair]
# matches_vector.pop(index_candidate_pair)

# print ("\tBest candidate pair for incremetal phase is (",image_A.id, ",", image_B.id, ") with ", len(matches_AB), "matches")
# src_pts  = np.around(np.float32([ image_A.keyPoints[m.queryIdx].pt for m in matches_AB ]).reshape(-1,2))
# src_desc = [ image_A.des[m.queryIdx] for m in matches_AB ]
# dst_pts  = np.around(np.float32([ image_B.keyPoints[m.trainIdx].pt for m in matches_AB ]).reshape(-1,2))
# dst_desc = [ image_B.des[m.queryIdx] for m in matches_AB ]


# ''' A verifier '''
# ## faire en sorte que l'image_A est l'image qui a déja etait utilisée, et l'image_B la nouvelle image ##
# assert image_A.points_2D_used != [] or image_B.points_2D_used != [], "Aucune des deux images n\'a été traitée auparavant (il faut améliorer le code)"
# assert not (image_A.points_2D_used != [] and image_B.points_2D_used != []), "les deux images ont été traitées auparavant (il faut améliorer le code)"
# if(image_B.points_2D_used != [] and image_A.points_2D_used == []):
#     # il faut les permuter
#     image_A, image_B = image_B, image_A
#     src_pts, dst_pts = dst_pts, src_pts
#     src_desc, dst_desc, dst_desc, src_desc          


# #############################################################
# ##           ''' Compute Fundamental Matrix '''            ##
# ##                 ''' Filter Matching '''                 ##
# #############################################################
# print ("\nCompute Fundamental Matrix")
# src_pts_inliers_F  = src_pts
# src_desc_inliers_F = src_desc
# dst_pts_inliers_F  = dst_pts
# dst_desc_inliers_F = dst_desc

# # _, maskInliers_F = Fundamentalmat.compute_FondamentalMatrix(src_pts = src_pts, dst_pts = dst_pts)
# # src_pts_inliers_F   = src_pts[maskInliers_F.ravel() == 1]
# # src_desc_inliers_F  = [ desc  for desc, i in zip (src_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
# # dst_pts_inliers_F   = dst_pts[maskInliers_F.ravel() == 1]
# # dst_desc_inliers_F  = [ desc  for desc, i in zip (dst_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]

# print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (FundamentalMatrix_inliers) ==", len(src_pts_inliers_F))


# ###########################################################
# ##           ''' Compute Essential Matrix '''            ##
# ##               ''' Filter Matching '''                 ##
# ###########################################################
# print ("\nCompute Essential Matrix")
# maskInliers_E = Essentialmat.compute_EssentialMatrix(src_pts = src_pts_inliers_F, dst_pts = dst_pts_inliers_F)
# # Update inliers points and descriptors
# src_pts_inliers_E   = src_pts_inliers_F[maskInliers_E.ravel() == 1]
# src_desc_inliers_E  = [ desc  for desc, i in zip (src_desc_inliers_F, np.arange(len(src_pts_inliers_F))) if maskInliers_E[i] == 1]
# dst_pts_inliers_E   = dst_pts_inliers_F[maskInliers_E.ravel() == 1]
# dst_desc_inliers_E  = [ desc  for desc, i in zip (dst_desc_inliers_F, np.arange(len(src_pts_inliers_F))) if maskInliers_E[i] == 1]
# print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (EssentialMatrix_inliers) ==", len(src_pts_inliers_E))


# ###########################################################
# ##              ''' Intersection 3D 2D '''               ##
# ##                ''' Filter Matching '''                ##
# ###########################################################
# pointIntersection, index_A, index_B = intersect2D(np.around(np.asarray(image_A.points_2D_used), decimals=0), np.around(np.asarray(src_pts_inliers_E), decimals=0)) 

# """ Retrieve points for solver (pt_A_3D_Previous & pt_B_2D_Current) """
# pt_A_2D_Previous = np.asarray(image_A.points_2D_used)[index_A, :].reshape(-1, 2)
# pt_A_3D_Previous = np.asarray(image_A.points_3D_used)[index_A, :].reshape(-1, 3)
# pt_A_2D_Current  = np.asarray(src_pts_inliers_E)[index_B, :].reshape(-1, 2)
# pt_B_2D_Current  = np.asarray(dst_pts_inliers_E)[index_B, :].reshape(-1, 2)
# # pt_B_desc_Current  = np.asarray(dst_pts_inliers_E)[index_B, :]  ## MANQUE

# """ Retrieve points for triangulate """
# New_points_A_src = np.delete(src_pts_inliers_E , index_B, axis=0).reshape(-1, 2)
# New_points_B_dst = np.delete(dst_pts_inliers_E , index_B, axis=0).reshape(-1, 2)

# print(" Le nombre de points 3D à projeter ", len(pointIntersection))


# ###########################################################
# ##      ''' Retrieve Transformation of Image_B '''       ##
# ###########################################################
# sizeInput = len(pt_A_3D_Previous)
# inliers_index = []
# if (not configuration["pnpsolver_method"]):
#     """Method 1""" # Solve PnP
#     _ , absolute_rVec, absolute_tVec = cv.solvePnP(pt_A_3D_Previous.reshape(-1, 1, 3), pt_B_2D_Current.reshape(-1, 1, 2), CameraMatrix, None, flags = cv.SOLVEPNP_ITERATIVE)  # All points, no inliers
    
# else:
#     print ("lami")
#     """Method 2""" # Solve PnP ransac ( par defaut )
#     # _, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(pt_A_3D_Previous.reshape(-1, 1, 3), pt_B_2D_Current.reshape(-1, 1, 2), CameraMatrix, None)

#     """Method 3""" # Solve PnP ransac paramétré
#     # _, absolute_rVecI, absolute_tVecI, inliers_index = cv.solvePnPRansac(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, useExtrinsicGuess = False, iterationsCount = 200000, reprojectionError = 3.0, flags = cv.SOLVEPNP_ITERATIVE)
#     # print (absolute_rVec, absolute_tVec, len(inliers_index))
#     # # _, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, useExtrinsicGuess = False, iterationsCount = 200000, reprojectionError = 3.0, flags = cv.SOLVEPNP_EPNP)
#     # # print (absolute_rVec, absolute_tVec, len(inliers_index))


#     # """ Remove inliers if solvePNPRansac is used """
#     # pt_A_2D_PreviousI = np.asarray(pt_A_2D_Previous)[inliers_index, :].reshape(-1, 2)
#     # pt_A_3D_PreviousI = np.asarray(pt_A_3D_Previous)[inliers_index, :].reshape(-1, 3)
#     # pt_A_2D_CurrentI  = np.asarray(pt_A_2D_Current)[inliers_index, :].reshape(-1, 2)
#     # pt_B_2D_CurrentI  = np.asarray(pt_B_2D_Current)[inliers_index, :].reshape(-1, 2)

#     _, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, useExtrinsicGuess = False, iterationsCount = 200000, reprojectionError = 3.0, flags = cv.SOLVEPNP_ITERATIVE)
#     print (absolute_rVec, absolute_tVec, len(inliers_index))
#     # _, absolute_rVec, absolute_tVec, inliers_index = cv.solvePnPRansac(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, useExtrinsicGuess = False, iterationsCount = 200000, reprojectionError = 5.0, flags = cv.SOLVEPNP_EPNP)
#     # print (absolute_rVec, absolute_tVec, len(inliers_index))


#     """ Remove inliers if solvePNPRansac is used """
#     pt_A_2D_Previous = np.asarray(pt_A_2D_Previous)[inliers_index, :].reshape(-1, 2)
#     pt_A_3D_Previous = np.asarray(pt_A_3D_Previous)[inliers_index, :].reshape(-1, 3)
#     pt_A_2D_Current  = np.asarray(pt_A_2D_Current)[inliers_index, :].reshape(-1, 2)
#     pt_B_2D_Current  = np.asarray(pt_B_2D_Current)[inliers_index, :].reshape(-1, 2)


# """Method 1""" # Solve PnP
# # _ , absolute_rVec, absolute_tVec = cv.solvePnP(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, flags = cv.SOLVEPNP_ITERATIVE)  # All points, no inliers
# # print (absolute_rVec, absolute_tVec)
# # _ , absolute_rVec, absolute_tVec = cv.solvePnP(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, flags = cv.SOLVEPNP_EPNP)  # All points, no inliers
# # print (absolute_rVec, absolute_tVec)
# # _ , absolute_rVec, absolute_tVec = cv.solvePnP(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, flags = cv.SOLVEPNP_DLS)  # All points, no inliers
# # print (absolute_rVec, absolute_tVec)
# # _ , absolute_rVec, absolute_tVec = cv.solvePnP(pt_A_3D_Previous.copy().reshape(-1, 1, 3), pt_B_2D_Current.copy().reshape(-1, 1, 2), CameraMatrix, None, flags = cv.SOLVEPNP_UPNP)  # All points, no inliers
# # print (absolute_rVec, absolute_tVec)


# """ Reprojection Error """
# project_points, _ = cv.projectPoints(pt_A_3D_Previous.T, absolute_rVec, absolute_tVec, CameraMatrix, distCoeffs=None, )
# reprojection_error = np.linalg.norm(pt_B_2D_Current - project_points.reshape(-1, 2)) / len(project_points)
# print ("\nJust Compute Reprojection Error")
# print("    Reprojection Error After Solve PnP ransac ( not default params ) --> ", reprojection_error, "with ", len(inliers_index), "inliers / ", sizeInput)


# # """ Reprojection Error """
# # project_points, _ = cv.projectPoints(pt_A_3D_Previous.T, absolute_rVecI, absolute_tVecI, CameraMatrix, distCoeffs=None, )
# # reprojection_error = np.linalg.norm(pt_B_2D_Current - project_points.reshape(-1, 2)) / len(project_points)
# # print ("\nJust Compute Reprojection Error")
# # print("    Reprojection Error After Solve PnP ransac ( not default params ) --> ", reprojection_error, "with ", len(inliers_index), "inliers / ", sizeInput)


# """ Retrieve Absolute Transformation of current image """
# absolute_Rt, _ = cv.Rodrigues(absolute_rVec)
# absolute_R_    = absolute_Rt.T

# proj_matrix = np.hstack((absolute_Rt, absolute_tVec))
# absolute_t_    = - absolute_R_ @ absolute_tVec
# image_B.setAbsolutePose(image_A, absolute_R_, absolute_t_)

# # """ Retrieve Relative Transformation of current image """  # ==> OSEF
# # transform_BC = current_relative_transform (image_B.absoluteTransformation["transform"], image_A.absoluteTransformation["transform"])
# # relative_R = transform_BC[:3, :3] 
# # relative_t = transform_BC[:3, 3]       
# # image_B.setRelativePose(image_A, relative_R, relative_t)


# ###########################################################
# ##    ''' Generate 3D_points using Triangulation '''     ##
# ###########################################################
# print ("\nGenerate 3D_points using Triangulation\n")
# points3d, index_to_Remove, = Essentialmat.Triangulate(image_A, image_B, New_points_A_src, New_points_B_dst, 1)
# # Update inliers points and descriptors
# points3d_F         = np.delete(points3d            , index_to_Remove, axis=0).tolist()
# print ("la nouvelle liste contient ",len(points3d_F), "points 3D")

# print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (Triangulate_inliers) ==", len(points3d_F))
# POINT_DRAW = points3d_F.copy()
# src_pts_inliers_F  = np.delete(New_points_A_src  , index_to_Remove, axis=0).reshape(-1, 2)
# # src_desc_inliers_F = np.delete(src_desc_inliers_RP , index_to_Remove, axis=0).tolist()  #MANQUE
# dst_pts_inliers_F  = np.delete(New_points_B_dst  , index_to_Remove, axis=0).reshape(-1, 2)
# # dst_desc_inliers_F = np.delete(dst_desc_inliers_RP , index_to_Remove, axis=0).tolist()  # Manque
# if (not len(points3d_F) == 0):
#     print ("Aucun nouveau point 3D à ajouter")
#     image_A.points_2D_used  = np.append(image_A.points_2D_used, src_pts_inliers_F, axis = 0)
#     # image_A.descriptor_used = np.append(image_A.descriptor_used, src_desc_inliers_F, axis = 0) Manque
#     image_A.points_3D_used  = np.append(image_A.points_3D_used, points3d_F, axis = 0)

#     image_B.points_2D_used  = np.array(pt_B_2D_Current.copy())
#     # image_B.descriptor_used = np.array(dst_desc_inliers_F.copy())# Manque
#     image_B.points_3D_used  = np.array(pt_A_3D_Previous.copy())

#     image_B.points_2D_used = np.append(image_B.points_2D_used, dst_pts_inliers_F, axis = 0)
#     # image_B.descriptor_used = np.append(image_A.descriptor_used, src_desc_inliers_F, axis = 0)  #MANQUE
#     image_B.points_3D_used = np.append(image_B.points_3D_used, points3d_F, axis = 0)


# # POINT_DRAW = np.append(POINT_DRAW, points3d_F, axis = 0)
# # POINT_DRAW = image_B.points_3D_used



###########################################################
##  ''' Recuprer tous les points 3D et les afficher '''  ##
###########################################################
print ("\n- Number of 3D points to draw is  ",len(POINT_DRAW), "points")

vis = o3d.visualization.Visualizer()
vis.create_window("Structure_from_Motion", 1280, 720)

# point_to_Draw = []

"""  Add Camera-pose of images """
scale = 1.
for i, img in enumerate(Images):
    if i == 2:
        pass
    else:
        mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scale).transform(img.absoluteTransformation["transform"])
        vis.add_geometry(mesh_img_i)
        # scale *= 1.5

    ### retrieve 3D-points to draw 
    # if (i == 0):
    #     point_to_Draw = img.points_3D_used
    # else :
    #     _, idx, idy = intersect2D(point_to_Draw, img.points_3D_used)
    #     toAdd =  np.delete(img.points_3D_used.copy() , idy, axis=0).tolist()
    #     # point_to_Draw = np.append(point_to_Draw, np.array(toAdd), axis = 0)


"""  Add point-Cloud of images """
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(POINT_DRAW).reshape(-1, 3))
vis.add_geometry(pcd)

"""  Launch Visualisation  """
vis.run()