###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################
from os import listdir
import os.path
from os import path
from numpy import loadtxt
from Utils import *

import json
import itertools

from config import *

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
print ("    ", configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])


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
    cv.imshow("matches_" + str(image_A.id)+"_"+ str(image_B.id), image_result)


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
_, maskInliers_F = Fundamentalmat.compute_FondamentalMatrix(src_pts = src_pts, dst_pts = dst_pts)
src_pts_inliers_F   = src_pts[maskInliers_F.ravel() == 1]
src_desc_inliers_F  = [ desc  for desc, i in zip (src_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
dst_pts_inliers_F   = dst_pts[maskInliers_F.ravel() == 1]
dst_desc_inliers_F  = [ desc  for desc, i in zip (dst_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
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
EssentialMat, maskInliers_E = Essentialmat.compute_EssentialMatrix(src_pts = src_pts_inliers_F, dst_pts = dst_pts_inliers_F)
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
points, Rot_RP, Transl_RP, mask_RP = Essentialmat.RecoverPose_3D_Points(EssentialMat = EssentialMat, src_pts = src_pts_inliers_E, dst_pts = dst_pts_inliers_E)

rVec, _ = cv.Rodrigues(Rot_RP.T)
tVec = np.dot(-Rot_RP.T,Transl_RP)

''' relative Pose '''
image_A.setRelativePose(image_A, np.eye(3, 3), np.zeros(3), np.zeros(3), np.zeros(3))
image_B.setRelativePose(image_A, Rot_RP, Transl_RP, rVec, tVec)

''' Absolute Pose '''
image_A.setAbsolutePose(np.eye(3, 3), np.ones(3), np.zeros(3), np.zeros(3))
image_B.setAbsolutePose(Rot_RP, Transl_RP, rVec, tVec)


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
points3d, index_to_Remove, = Essentialmat.Triangulate(Rot_RP, Transl_RP, src_pts_inliers_RP, dst_pts_inliers_RP)

print ("Rotation de l'image 2 par rapport à l'mage 1 \n", Rot_RP)
print ("Translation de l'image 2 par rapport à l'image 1\n", Transl_RP)

# Update inliers points and descriptors
points3d_F         = np.delete(points3d            , index_to_Remove, axis=0).tolist()
src_pts_inliers_F  = np.delete(src_pts_inliers_RP  , index_to_Remove, axis=0).tolist()
src_desc_inliers_F = np.delete(src_desc_inliers_RP , index_to_Remove, axis=0).tolist()
dst_pts_inliers_F  = np.delete(dst_pts_inliers_RP  , index_to_Remove, axis=0).tolist()
dst_desc_inliers_F = np.delete(dst_desc_inliers_RP , index_to_Remove, axis=0).tolist()
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (recoverPose_inliers) ==", points)

image_A.points_2D_used  = src_pts_inliers_F
image_A.descriptor_used = src_desc_inliers_F
image_A.points_3D_used  = points3d_F

image_B.points_2D_used  = dst_pts_inliers_F
image_B.descriptor_used = dst_desc_inliers_F
image_B.points_3D_used  = points3d_F

POINT_DRAW = points3d_F



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
_, maskInliers_F = Fundamentalmat.compute_FondamentalMatrix(src_pts = src_pts, dst_pts = dst_pts)
src_pts_inliers_F   = src_pts[maskInliers_F.ravel() == 1]
src_desc_inliers_F  = [ desc  for desc, i in zip (src_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
dst_pts_inliers_F   = dst_pts[maskInliers_F.ravel() == 1]
dst_desc_inliers_F  = [ desc  for desc, i in zip (dst_desc, np.arange(len(src_pts))) if maskInliers_F[i] == 1]
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (FundamentalMatrix_inliers) ==", len(src_pts_inliers_F))


###########################################################
##              ''' Intersection 3D 2D '''               ##
##                ''' Filter Matching '''                ##
###########################################################
pointIntersection, index_A, index_B = intersect2D(np.around(np.asarray(image_A.points_2D_used), decimals=0), np.around(np.asarray(src_pts_inliers_F), decimals=0)) 

pt_1  = [image_A.points_2D_used[i] for i in index_A]
pt_3D = [image_A.points_3D_used[i] for i in index_A]
pt_2  = [src_pts_inliers_F[i]      for i in index_B]
 
New_points_2DTO_3D_src = np.delete(src_pts_inliers_F , index_B, axis=0)
New_points_2DTO_3D_dst = np.delete(dst_pts_inliers_F , index_B, axis=0)

print(" Le nombre de points 3D à projeter ", len(pointIntersection))


###########################################################
##      ''' Retrieve Transformation of Image_B '''       ##
###########################################################
_, absolute_rVec, absolute_tVec = cv.solvePnP(np.array(pt_3D).reshape(-1, 1, 3), np.array(pt_2).reshape(-1, 1, 2), CameraMatrix, None)

""" Reprojection Error """
# p1, _ = cv.projectPoints(np.array(pt_3D).T, absolute_RVec, absolute_TVec, CameraMatrix, distCoeffs=None)
# reprojection_error1 = np.linalg.norm(np.array(pt_2).reshape(-1, 2) - p1.reshape(-1, 2)) / len(p1)
# print("    \nReprojection Error Image_A --> ", reprojection_error1)

absolute_Rt, _ = cv.Rodrigues(absolute_rVec)
absolute_R_ = np.asarray(absolute_Rt.T)
absolute_t_ = -np.dot(np.linalg.inv(absolute_Rt) , np.asarray(absolute_tVec))

relative_tVec = absolute_tVec - image_A.absoluteTransformation["tVec"]
relative_rVec = None

relative_R = np.linalg.inv(image_A.absoluteTransformation["rotation"]) @ absolute_R_
relative_t = - np.linalg.inv(relative_R.T) @ np.asarray(relative_tVec)

''' relative Pose '''
# image_A.setRelativePose(image_A, np.eye(3, 3), np.zeros(3), np.zeros(3), np.zeros(3))
image_B.setRelativePose(image_A, relative_R, relative_t, relative_rVec, relative_tVec)

''' Absolute Pose '''
# image_A.setAbsolutePose(np.eye(3, 3), np.ones(3), np.zeros(3), np.zeros(3))
image_B.setAbsolutePose(absolute_R_, absolute_t_, absolute_rVec, absolute_tVec)


print ("Rotation de l'image 3 par rapport à un référenciel    (Image 1) \n", absolute_R_)
print ("translation de l'image 3 par rapport à un référenciel (Image 1)\n", absolute_t_)


###########################################################
##    ''' Generate 3D_points using Triangulation '''     ##
###########################################################
print ("\nGenerate 3D_points using Triangulation\n")
points3d, index_to_Remove, = Essentialmat.Triangulate(relative_R, relative_t, New_points_2DTO_3D_src, New_points_2DTO_3D_dst)

# # JUST TEST #
# rvec2, _ = cv.Rodrigues(Rot_RP.T)
# tvec2 = np.dot(-Rot_RP.T,Transl_RP)

# p1, _ = cv.projectPoints(np.asarray(points3d).T, rvec2, tvec2, CameraMatrix, distCoeffs=None)
# reprojection_error1 = np.linalg.norm(np.array(dst_pts_inliers_RP).reshape(-1, 2) - p1.reshape(-1, 2)) / len(p1)
# print("    \nReprojection Error Image_BB --> ", reprojection_error1)
# # END TEST #

# Update inliers points and descriptors
points3d_F         = np.delete(points3d            , index_to_Remove, axis=0).tolist()
src_pts_inliers_F  = np.delete(src_pts_inliers_RP  , index_to_Remove, axis=0).tolist()
src_desc_inliers_F = np.delete(src_desc_inliers_RP , index_to_Remove, axis=0).tolist()
dst_pts_inliers_F  = np.delete(dst_pts_inliers_RP  , index_to_Remove, axis=0).tolist()
dst_desc_inliers_F = np.delete(dst_desc_inliers_RP , index_to_Remove, axis=0).tolist()

image_A.points_2D_used = np.append(image_A.points_2D_used, src_pts_inliers_F, axis = 0)
image_A.descriptor_used = np.append(image_A.descriptor_used, src_desc_inliers_F, axis = 0)
image_A.points_3D_used = np.append(image_A.points_3D_used, points3d_F, axis = 0)

image_B.points_2D_used  = dst_pts_inliers_F
image_B.descriptor_used = dst_desc_inliers_F
image_B.points_3D_used  = pt_3D
image_A.points_3D_used = np.append(image_A.points_3D_used, points3d_F, axis = 0)


POINT_DRAW = np.append(POINT_DRAW, points3d_F, axis = 0)
###########################################################
##  ''' Recuprer tous les points 3D et les afficher '''  ##
###########################################################
print ("la nouvelle liste contient ",len(POINT_DRAW), "points 3D")

import open3d as o3d
vis = o3d.visualization.Visualizer()
vis.create_window()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(POINT_DRAW)
vis.add_geometry(pcd)

vis.run()