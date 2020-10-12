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
# from FundamentalMatEstimation import *
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
ImageFolder = "Imgs/Saint_Roch_Original/"
''' toutes les images ont la même résolution et sont prises par mon samsung S7 '''
CameraMatrix = loadtxt(ImageFolder+"samsung-s7-1920x1080.csv", delimiter=',')  #FullHD

''' toutes les images ont la même résolution et sont prises par mon samsung S7 '''
DistortionCoef = None  # not used




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''          PHASE One: prepare Data           '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

assert path.exists(ImageFolder), 'veuillez verifier le chemin du Folder'
images_name = sorted([file for file in listdir(ImageFolder) if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".png")])


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




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''      PHASE Three: Incremental SfM Data     '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
              
###############################################################
## ''' Retrieve best candidate pair for incremetal phase ''' ##
###############################################################
print ("\nRetrieve best candidate pair for incremetal phase")
index_candidate_pair = np.argmax([len (x.matches) for x in matches_vector])
matching_AB = matches_vector[index_candidate_pair]
matches_vector.pop(index_candidate_pair)
print ("\nRetrieve best candidate pair for initialization: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")

print (len(matching_AB.image_A.points_2D_used))
print (len(matching_AB.image_B.points_2D_used))

''' A verifier '''
## Rectify precedent_image, current_image
## faire en sorte que l'image_A est l'image qui a déja etait utilisée, et l'image_B la nouvelle image ##
assert matching_AB.image_A.points_2D_used != [] or matching_AB.image_B.points_2D_used != [], "Aucune des deux images n\'a été traitée auparavant (il faut améliorer le code)"
assert not (matching_AB.image_A.points_2D_used != [] and matching_AB.image_B.points_2D_used != []), "les deux images ont été traitées auparavant (il faut améliorer le code)"
if(matching_AB.image_B.points_2D_used != [] and matching_AB.image_A.points_2D_used == []):
    # il faut les permuter
    matching_AB.image_A, matching_AB.image_B = matching_AB.image_B, matching_AB.image_A
    matching_AB.src_pts, matching_AB.dst_pts = matching_AB.dst_pts, matching_AB.src_pts
    matching_AB.src_desc, matching_AB.dst_desc, matching_AB.dst_desc, matching_AB.src_desc          


################################################################
##      ''' Compute Transformation Matrix using PNP '''       ##
##      ''' Triangulation and incerement point-Cloud '''      ##
################################################################
print ("Compute transformation and initialize point-Cloud")
matching_AB.computePose_3D2D(Essentialmat)




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''       PHASE Four: Display Point-Cloud      '''''''''          
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
###########################################################
##  ''' Recuprer tous les points 3D et les afficher '''  ##
###########################################################
### retrieve 3D-points to draw 
p_cloud_list = [img.points_3D_used for img in Images]
p_cloud = union(p_cloud_list)
print ("\n- Number of 3D points to draw is  ",len(p_cloud), "points")

vis = o3d.visualization.Visualizer()
vis.create_window("Structure_from_Motion", 1280, 720)

# point_to_Draw = []

"""  Add Camera-pose of images """
scale = 1.
for i, img in enumerate(Images):
    mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scale).transform(img.absoluteTransformation["transform"])
    vis.add_geometry(mesh_img_i)
    scale *= 0.5

"""  Add point-Cloud of images """
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(p_cloud).reshape(-1, 3))
vis.add_geometry(pcd)

"""  Launch Visualisation  """
vis.run()