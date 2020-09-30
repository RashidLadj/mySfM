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


ImageFolder = "Imgs/"
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
##   ''' Save keypoint and Descriptor in Json File  '''  ##
###########################################################
# Features_data = {}
# Features_data['FeaturePoints'] = []
# for image in Images:
#     Features_data['FeaturePoints'].append({
#         'ID_img': image.id,
#         'features': [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (image.points, image.des)]
#     })

# if not path.exists("Json_Files/"):
#     os.mkdir("Json_Files/")

# with open('Json_Files/Features_data.json', 'w', encoding='utf-8') as outfile:
#     json.dump(Features_data, outfile)


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
#
# Matching_data = {}
# Matching_data['MatchingPoints'] = []
#
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
image_A, image_B, matches_AB = matches_vector[index_candidate_pair]
matches_vector.pop(index_candidate_pair)
print ("\tBest candidate pair for initialization is (",image_A.id, ",", image_B.id, ") with ", len(matches_AB), "matches")
src_pts  = np.around(np.float32([ image_A.keyPoints[m.queryIdx].pt for m in matches_AB ]).reshape(-1,2))
src_desc = [ image_A.des[m.queryIdx] for m in matches_AB ]
dst_pts  = np.around(np.float32([ image_B.keyPoints[m.trainIdx].pt for m in matches_AB ]).reshape(-1,2))
dst_desc = [ image_B.des[m.trainIdx] for m in matches_AB ]   

###################################################################
## ''' Save Matching (Keypoint and descriptor used)Json File ''' ##
###################################################################
# points_A = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (src_pts, src_desc)]
# points_B = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (dst_pts, dst_desc)]

# Matching_data['MatchingPoints'].append({
#     'ID_Matching': "match_" + image_A.id + "_" + image_B.id,
#     'number_matches': len(matches),
#     'matches': [ {"ID_img": image_A.id, "features_Points" : points_A} , {"ID_img": image_B.id, "features_Points" : points_B} ]
# })

# if not path.exists("Json_Files/"):
#     os.mkdir("Json_Files/")

# with open('Json_Files/Matching_data.json', 'w', encoding='utf-8') as outfile:
#     json.dump(Matching_data, outfile)            


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
# print ("\nCamera Matrix : \n", CameraMatrix)


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

# points_A = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (src_pts, src_desc)]
# points_B = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (dst_pts, dst_desc)]

# Matching_data['MatchingPoints_OptimizeEssensial'].append({
#     'ID_Matching': "match_" + image_A.id + "_" + image_B.id,
#     'number_matches': len(matches),
#     'matches': [ {"ID_img": image_A.id, "features_Points" : points_A} , {"ID_img": image_B.id, "features_Points" : points_B} ]
# })

# if not path.exists("Json_Files/"):
#     os.mkdir("Json_Files/")

# with open('Json_Files/Matching_data.json', 'w', encoding='utf-8') as outfile:
#     json.dump(Matching_data, outfile)        


###########################################################
##      ''' Retrieve Transformation of Image_B '''       ##
###########################################################
print ("\nRetrieve Transformation of Image_B")
points, Rot_RP, Transl_RP, mask_RP = Essentialmat.RecoverPose_3D_Points(EssentialMat = EssentialMat, src_pts = src_pts_inliers_E, dst_pts = dst_pts_inliers_E)
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
print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (recoverPose_inliers) ==", points)

image_A.points_2D_used  = src_pts_inliers_F
image_A.descriptor_used = src_desc_inliers_F
image_A.points_3D_used  = points3d_F

image_B.points_2D_used  = dst_pts_inliers_F
image_B.descriptor_used = dst_desc_inliers_F
image_B.points_3D_used  = points3d_F

###########################################################
##       ''' Config params of Essential Matrix '''       ##
###########################################################
# for data in Matching_data['MatchingPoints']:
#     print (data['ID_Matching'])
#     [print ("\t" + image['ID_img']) for image in data['matches']]

#     # matches_ = data['matches']

# [print (data['number_matches'])   for data in Matching_data['MatchingPoints']



''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''                INCREMENTAL PHASE                 '''               
''''''''''''''''''''''''''''''''''''''''''''''''''''''''                 
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


###################################################################
## ''' Save Matching (Keypoint and descriptor used)Json File ''' ##
###################################################################
    # points_A = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (src_pts, src_desc)]
    # points_B = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (dst_pts, dst_desc)]

    # Matching_data['MatchingPoints'].append({
    #     'ID_Matching': "match_" + image_A.id + "_" + image_B.id,
    #     'number_matches': len(matches),
    #     'matches': [ {"ID_img": image_A.id, "features_Points" : points_A} , {"ID_img": image_B.id, "features_Points" : points_B} ]
    # })

    # if not path.exists("Json_Files/"):
    #     os.mkdir("Json_Files/")

    # with open('Json_Files/Matching_data.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(Matching_data, outfile)            


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
''' je dois garder les indices dans l'ordre '''
pt_1  = [image_A.points_2D_used[i] for i in index_A]
pt_3D = [image_A.points_3D_used[i] for i in index_A]
pt_2  = [src_pts_inliers_F[i]      for i in index_B]

print(" Le nombre de points 3D à projeter ", len(pointIntersection))
# [print (a, b, c) for a,b,c in zip(pt_1, pt_2, pt_3D)]

###########################################################
##      ''' Retrieve Transformation of Image_B '''       ##
###########################################################
_, rVec, tVec = cv.solvePnP(np.array(pt_3D).reshape(-1, 1, 3), np.array(pt_2).reshape(-1, 1, 2), CameraMatrix, None)

p1, _ = cv.projectPoints(np.array(pt_3D).T, rVec, tVec, CameraMatrix, distCoeffs=None)
reprojection_error1 = np.linalg.norm(np.array(pt_2).reshape(-1, 2) - p1.reshape(-1, 2)) / len(p1)
print("    \nReprojection Error Image_A --> ", reprojection_error1)


Rt, _ = cv.Rodrigues(rVec)
R_ = np.asarray(Rt.T)
print ("Rotation Image 3  --> \n", R_)
t_ = -np.dot(np.linalg.inv(Rt) , np.asarray(tVec))
print ("Rotation Image 3  --> \n", t_)

