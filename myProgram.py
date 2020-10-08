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


###########################################################
##            ''' Load Configuration File '''            ##
###########################################################
configuration = default_config()


ImageFolder = "Images/"
assert path.exists(ImageFolder), 'veuillez verifier le chemin du Folder'
images_name = sorted([f for f in listdir(ImageFolder)])


###########################################################
##                  ''' Create Images '''                ##
###########################################################
print ("Read images")
Images = []
for image in images_name:
    Images.append(Image(ImageFolder, image, configuration["feature_process_size"]))
print ("    Images = ", images_name)


###########################################################
##          ''' Define Detector and Matcher '''          ##
###########################################################
detector, matcher= keyPointDetectorAndMatchingParams(keyPointDetectorMethod = configuration["feature_type"], matcherMethod = configuration["matcher_type"], crosscheck = configuration["matcher_crosscheck"], Max_keyPoint = configuration["max_keyPoint"])      
assert not detector == None and not matcher == None, "Problems !!!"
print ("\nDetector == ", type(detector), " Matcher = ", type(matcher))


###########################################################
##       ''' Calculate KeyPoint and Descriptor '''       ##
###########################################################
print ("\nDetect and compute descriptors")
for image in Images:
    image.keyPointDetector(detector)
    print ('  size of descriptors of', image.id, image.des.shape)
    

###########################################################
##             ''' Matching configuration '''            ##
###########################################################
print ("\nMatching configuration")
matching = Matching(matcher, configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])
# print ("    ", configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])


###########################################################
##  ''' Compute Matching beetwin each pair of image '''  ##
###########################################################
print ("\nMatching points beetwin each pair of image")

image_Pair = list(itertools.combinations(Images, 2))

matches_vector = []
for image_A, image_B in image_Pair:
    matches, image_result = matching.match(image_A, image_B)
    matches_vector.append([image_A, image_B, matches])
    print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (No optimize) ==", len(matches))
    # cv.imshow("matches_" + str(image_A.id)+"_"+ str(image_B.id), image_result)


###############################################################
##  ''' Retrieve best candidate pair for initialization '''  ##
###############################################################
print ("\nRetrieve best candidate pair for initialization")

index_candidate_pair = np.argmax([len (x[2]) for x in matches_vector])
index_candidate_pair = 0                            # temporary value 
image_A, image_B, matches_AB = matches_vector[index_candidate_pair]
matches_vector.pop(index_candidate_pair)
print ("\tBest candidate pair for initialization is (",image_A.id, ",", image_B.id, ") with ", len(matches_AB), "matches")
src_pts  = np.around(np.float32([ image_A.keyPoints[m.queryIdx].pt for m in matches_AB ]).reshape(-1, 2))
src_desc = [ image_A.des[m.queryIdx] for m in matches_AB ]
dst_pts  = np.around(np.float32([ image_B.keyPoints[m.trainIdx].pt for m in matches_AB ]).reshape(-1, 2))
dst_desc = [ image_B.des[m.trainIdx] for m in matches_AB ]            


###########################################################
##        ''' Load Camera Matrix and update it '''       ##
##                Sumsung S7 (1920 x 1080)               ##
###########################################################
''' toutes les images ont la même résolution et sont prises par mon sumsung S7 '''
CameraMatrix = loadtxt('CameraMatrixArray.csv', delimiter=',')  #FullHD
ComputeCameraMatrix(CameraMatrix, Images[0])
# print ("\nCamera Matrix : \n", CameraMatrix)


#######################################################
##        ''' Load distortion coefficients '''       ##
##              Sumsung S7 (1920 x 1080)             ##
#######################################################
''' toutes les images ont la même résolution et sont prises par mon sumsung S7 '''
DistortionCoef = None


#############################################################
##       ''' Config params of Fundamental Matrix '''       ##
#############################################################
# print ("\nEssential Matrix Configuration\n")
Fundamentalmat = FundamentalMatrix(methodOptimizer = configuration["fundamental_method"])


#############################################################
##           ''' Compute Fundamental Matrix '''            ##
##                 ''' Filter Matching '''                 ##
#############################################################
print ("\nCompute Fundamental Matrix")
src_pts_inliers_F  = src_pts
src_desc_inliers_F = src_desc
dst_pts_inliers_F  = dst_pts
dst_desc_inliers_F = dst_desc

# _, maskInliers_F = Fundamentalmat.compute_FondamentalMatrix(src_pts = src_pts, dst_pts = dst_pts)
# src_pts_inliers_F   = src_pts[maskInliers_F.ravel() == 1]
# src_desc_inliers_F  = [ desc  for desc, i in zip (src_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
# dst_pts_inliers_F   = dst_pts[maskInliers_F.ravel() == 1]
# dst_desc_inliers_F  = [ desc  for desc, i in zip (dst_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]

print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (FundamentalMatrix_inliers) ==", len(src_pts_inliers_F))

###########################################################
##       ''' Config params of Essential Matrix '''       ##
###########################################################
# print ("\nEssential Matrix Configuration\n")
Essentialmat = EssentialMatrix(methodOptimizer = configuration["essential_method"], CameraMatrixArray = CameraMatrix)


###########################################################
##           ''' Compute Essential Matrix '''            ##
##               ''' Filter Matching '''                 ##
###########################################################
print ("\nCompute Essential Matrix")
maskInliers_E = Essentialmat.compute_EssentialMatrix(src_pts = src_pts_inliers_F, dst_pts = dst_pts_inliers_F)
# Update inliers points and descriptors
src_pts_inliers_E   = src_pts_inliers_F[maskInliers_E.ravel() == 1]
src_desc_inliers_E  = [ desc  for desc, i in zip (src_desc_inliers_F, np.arange(len(src_pts_inliers_F))) if maskInliers_E[i] == 1]
dst_pts_inliers_E   = dst_pts_inliers_F[maskInliers_E.ravel() == 1]
dst_desc_inliers_E  = [ desc  for desc, i in zip (dst_desc_inliers_F, np.arange(len(src_pts_inliers_F))) if maskInliers_E[i] == 1]
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (EssentialMatrix_inliers) ==", len(src_pts_inliers_E))


###########################################################
##      ''' Retrieve Transformation of Image_B '''       ##

###########################################################
print ("\nRetrieve Transformation of Image_B")
points, Rot_RP, Transl_RP, mask_RP = Essentialmat.RecoverPose_3D_Points(src_pts = src_pts_inliers_E, dst_pts = dst_pts_inliers_E)

# ''' relative Pose '''
# image_A.setRelativePose(image_A, np.eye(3, 3), np.zeros(3))
# image_B.setRelativePose(image_A, Rot_RP, Transl_RP)

''' Absolute Pose '''
image_A.setAbsolutePose(np.eye(3, 3), np.zeros(3))
image_B.setAbsolutePose(Rot_RP, Transl_RP)

# Update inliers points and descriptors
src_pts_inliers_RP   = src_pts_inliers_E[mask_RP.ravel() == 255]
src_desc_inliers_RP  = [ desc  for desc, i in zip (src_desc_inliers_E, np.arange(len(src_pts_inliers_E))) if mask_RP[i] == 255]
dst_pts_inliers_RP   = dst_pts_inliers_E[mask_RP.ravel() == 255]
dst_desc_inliers_RP  = [ desc  for desc, i in zip (dst_desc_inliers_E, np.arange(len(src_pts_inliers_E))) if mask_RP[i] == 255]
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (recoverPose_inliers) ==", points)


###########################################################
##    ''' Generate 3D_points using Triangulation '''     ##
###########################################################
print ("\nGenerate 3D_points using Triangulation\n")
points3d, index_to_Remove, = Essentialmat.Triangulate(image_A, image_B, src_pts_inliers_RP, dst_pts_inliers_RP)

# Update inliers points and descriptors
points3d_F         = np.delete(points3d            , index_to_Remove, axis=0).tolist()
src_pts_inliers_F  = np.delete(src_pts_inliers_RP  , index_to_Remove, axis=0).tolist()
src_desc_inliers_F = np.delete(src_desc_inliers_RP , index_to_Remove, axis=0).tolist()
dst_pts_inliers_F  = np.delete(dst_pts_inliers_RP  , index_to_Remove, axis=0).tolist()
dst_desc_inliers_F = np.delete(dst_desc_inliers_RP , index_to_Remove, axis=0).tolist()
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (Triangulate_inliers) ==", len(points3d_F))

image_A.points_2D_used  = src_pts_inliers_F
image_A.descriptor_used = src_desc_inliers_F
image_A.points_3D_used  = points3d_F

image_B.points_2D_used  = dst_pts_inliers_F
image_B.descriptor_used = dst_desc_inliers_F
image_B.points_3D_used  = points3d_F

POINT_DRAW = image_A.points_3D_used




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''             INCREMENTAL PHASE              '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
              
###############################################################
##  ''' Retrieve best candidate pair for incremetal phase '''  ##
###############################################################
print ("\nRetrieve best candidate pair for incremetal phase")

index_candidate_pair = np.argmax([len (x[2]) for x in matches_vector])
image_A, image_B, matches_AB = matches_vector[index_candidate_pair]
matches_vector.pop(index_candidate_pair)

print ("\tBest candidate pair for incremetal phase is (",image_A.id, ",", image_B.id, ") with ", len(matches_AB), "matches")
src_pts  = np.around(np.float32([ image_A.keyPoints[m.queryIdx].pt for m in matches_AB ]).reshape(-1,2))
src_desc = [ image_A.des[m.queryIdx] for m in matches_AB ]
dst_pts  = np.around(np.float32([ image_B.keyPoints[m.trainIdx].pt for m in matches_AB ]).reshape(-1,2))
dst_desc = [ image_B.des[m.queryIdx] for m in matches_AB ]


''' A verifier '''
## faire en sorte que l'image_A est l'image qui a déja etait utilisée, et l'image_B la nouvelle image ##
assert image_A.points_2D_used != [] or image_B.points_2D_used != [], "Aucune des deux images n\'a été traitée auparavant (il faut améliorer le code)"
assert not (image_A.points_2D_used != [] and image_B.points_2D_used != []), "les deux images ont été traitées auparavant (il faut améliorer le code)"
if(image_B.points_2D_used != [] and image_A.points_2D_used == []):
    # il faut les permuter
    image_A, image_B = image_B, image_A
    src_pts, dst_pts = dst_pts, src_pts
    src_desc, dst_desc, dst_desc, src_desc          


#############################################################
##           ''' Compute Fundamental Matrix '''            ##
##                 ''' Filter Matching '''                 ##
#############################################################
print ("\nCompute Fundamental Matrix")
src_pts_inliers_F  = src_pts
src_desc_inliers_F = src_desc
dst_pts_inliers_F  = dst_pts
dst_desc_inliers_F = dst_desc

# _, maskInliers_F = Fundamentalmat.compute_FondamentalMatrix(src_pts = src_pts, dst_pts = dst_pts)
# src_pts_inliers_F   = src_pts[maskInliers_F.ravel() == 1]
# src_desc_inliers_F  = [ desc  for desc, i in zip (src_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
# dst_pts_inliers_F   = dst_pts[maskInliers_F.ravel() == 1]
# dst_desc_inliers_F  = [ desc  for desc, i in zip (dst_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]

print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (FundamentalMatrix_inliers) ==", len(src_pts_inliers_F))


###########################################################
##           ''' Compute Essential Matrix '''            ##
##               ''' Filter Matching '''                 ##
###########################################################
print ("\nCompute Essential Matrix")
maskInliers_E = Essentialmat.compute_EssentialMatrix(src_pts = src_pts_inliers_F, dst_pts = dst_pts_inliers_F)
# Update inliers points and descriptors
src_pts_inliers_E   = src_pts_inliers_F[maskInliers_E.ravel() == 1]
src_desc_inliers_E  = [ desc  for desc, i in zip (src_desc_inliers_F, np.arange(len(src_pts_inliers_F))) if maskInliers_E[i] == 1]
dst_pts_inliers_E   = dst_pts_inliers_F[maskInliers_E.ravel() == 1]
dst_desc_inliers_E  = [ desc  for desc, i in zip (dst_desc_inliers_F, np.arange(len(src_pts_inliers_F))) if maskInliers_E[i] == 1]
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (EssentialMatrix_inliers) ==", len(src_pts_inliers_E))


###########################################################
##              ''' Intersection 3D 2D '''               ##
##                ''' Filter Matching '''                ##
###########################################################
pointIntersection, index_A, index_B = intersect2D(np.around(np.asarray(image_A.points_2D_used), decimals=0), np.around(np.asarray(src_pts_inliers_E), decimals=0)) 

""" Retrieve points for solver (pt_A_3D_Previous & pt_B_2D_Current) """
pt_A_2D_Previous = np.asarray(image_A.points_2D_used)[index_A, :]
pt_A_3D_Previous = np.asarray(image_A.points_3D_used)[index_A, :]
pt_A_2D_Current  = np.asarray(src_pts_inliers_E)[index_B, :]
pt_B_2D_Current  = np.asarray(dst_pts_inliers_E)[index_B, :]
# pt_B_desc_Current  = np.asarray(dst_pts_inliers_E)[index_B, :]  ## MANQUE

""" Retrieve points for triangulate """
New_points_A_src = np.delete(src_pts_inliers_E , index_B, axis=0)
New_points_B_dst = np.delete(dst_pts_inliers_E , index_B, axis=0)

print(" Le nombre de points 3D à projeter ", len(pointIntersection))


###########################################################
##      ''' Retrieve Transformation of Image_B '''       ##
###########################################################
"""Method 1""" # Solve PnP
# _ , absolute_rVec, absolute_tVec = cv.solvePnP(pt_A_3D_Previous.reshape(-1, 1, 3), pt_B_2D_Current.reshape(-1, 1, 2), CameraMatrix, None)  # All points, no inliers

"""Method 2""" # Solve PnP ransac ( par defaut )
# _, absolute_rVec, absolute_tVec, inlierss = cv.solvePnPRansac(pt_A_3D_Previous.reshape(-1, 1, 3), pt_B_2D_Current.reshape(-1, 1, 2), CameraMatrix, None)

"""Method 3""" # Solve PnP ransac paramétré
_, absolute_rVec, absolute_tVec, inlierss = cv.solvePnPRansac(pt_A_3D_Previous.reshape(-1, 1, 3), pt_B_2D_Current.reshape(-1, 1, 2), CameraMatrix, None, useExtrinsicGuess = False, iterationsCount = 20000, reprojectionError = 3.0, flags = cv.SOLVEPNP_EPNP)

"""Remove inliers if solvePNPRansac is used"""
pt_A_3D_Previous_inliers = np.asarray(pt_A_3D_Previous)[inlierss, :].reshape(-1, 3)
pt_B_2D_Current_inliers = np.asarray(pt_B_2D_Current)[inlierss, :].reshape(-1, 2)

""" Reprojection Error """

project_points, _ = cv.projectPoints(pt_A_3D_Previous_inliers.T, absolute_rVec, absolute_tVec, CameraMatrix, distCoeffs=None)
reprojection_error = np.linalg.norm(pt_B_2D_Current_inliers.reshape(-1, 2) - project_points.reshape(-1, 2)) / len(project_points)
print ("\nJust Compute Reprojection Error")
print("    Reprojection Error After Solve PnP ransac ( not default params ) --> ", reprojection_error, "with ", len(inlierss), "inliers / ", len(pt_A_3D_Previous))

""" Retrieve points for solver (pt_A_3D_Previous & pt_B_2D_Current) """
pt_A_2D_Previous = np.asarray(image_A.points_2D_used)[inlierss, :]
pt_A_3D_Previous = np.asarray(image_A.points_3D_used)[inlierss, :]
pt_A_2D_Current  = np.asarray(src_pts_inliers_E)[inlierss, :]
pt_B_2D_Current  = np.asarray(dst_pts_inliers_E)[inlierss, :]

""" Retrieve Absolute Transformation of current image """
absolute_Rt, _ = cv.Rodrigues(absolute_rVec)
absolute_R_    = np.asarray(absolute_Rt.T)
absolute_t_    = - absolute_R_ @ np.asarray(absolute_tVec)
image_B.setAbsolutePose(absolute_R_, absolute_t_)

# """ Retrieve Relative Transformation of current image """  # ==> OSEF
# transform_BC = current_relative_transform (image_B.absoluteTransformation["transform"], image_A.absoluteTransformation["transform"])
# relative_R = transform_BC[:3, :3] 
# relative_t = transform_BC[:3, 3]       
# image_B.setRelativePose(image_A, relative_R, relative_t)


###########################################################
##    ''' Generate 3D_points using Triangulation '''     ##
###########################################################
print ("\nGenerate 3D_points using Triangulation\n")
points3d, index_to_Remove, = Essentialmat.Triangulate(image_A, image_B, New_points_A_src, New_points_B_dst)
# Update inliers points and descriptors
points3d_F         = np.delete(points3d            , index_to_Remove, axis=0).tolist()
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (Triangulate_inliers) ==", len(points3d_F))

src_pts_inliers_F  = np.delete(New_points_A_src  , index_to_Remove, axis=0).tolist()
src_desc_inliers_F = np.delete(src_desc_inliers_RP , index_to_Remove, axis=0).tolist()
dst_pts_inliers_F  = np.delete(New_points_B_dst  , index_to_Remove, axis=0).tolist()
dst_desc_inliers_F = np.delete(dst_desc_inliers_RP , index_to_Remove, axis=0).tolist()

image_A.points_2D_used = np.append(image_A.points_2D_used, src_pts_inliers_F, axis = 0)
image_A.descriptor_used = np.append(image_A.descriptor_used, src_desc_inliers_F, axis = 0)
image_A.points_3D_used = np.append(image_A.points_3D_used, points3d_F, axis = 0)

image_B.points_2D_used  = pt_B_2D_Current
image_B.descriptor_used = dst_desc_inliers_F
image_B.points_3D_used  = pt_A_3D_Previous

image_B.points_2D_used = np.append(image_B.points_2D_used, dst_pts_inliers_F, axis = 0)
# image_B.descriptor_used = np.append(image_A.descriptor_used, src_desc_inliers_F, axis = 0)  #MANQUE
image_B.points_3D_used = np.append(image_B.points_3D_used, points3d_F, axis = 0)

POINT_DRAW = np.append(POINT_DRAW, points3d_F, axis = 0)
POINT_DRAW = image_B.points_3D_used

###########################################################
##  ''' Recuprer tous les points 3D et les afficher '''  ##
###########################################################
print ("la nouvelle liste contient ",len(POINT_DRAW), "points 3D")

vis = o3d.visualization.Visualizer()
vis.create_window("Structure_from_Motion", 1280, 720)

# point_to_Draw = []

"""  Add Camera-pose of images """
scale = 1.
for i, img in enumerate(Images):
    mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scale).transform(img.absoluteTransformation["transform"])
    vis.add_geometry(mesh_img_i)
    scale *= .5

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