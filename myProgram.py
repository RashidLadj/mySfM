###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################
from os import listdir
import os.path
from os import path

import json
import itertools

from config import *

from Image import *
from DescriptorMatcherConfig import *
from Matching import *


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
##   ''' Save keypoint and Descriptor in Json File  '''  ##
###########################################################
Features_data = {}
Features_data['FeaturePoints'] = []
for image in Images:
    Features_data['FeaturePoints'].append({
        'ID_img': image.id,
        'features': [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (image.points, image.des)]
    })

if not path.exists("Json_Files/"):
    os.mkdir("Json_Files/")

with open('Json_Files/Features_data.json', 'w', encoding='utf-8') as outfile:
    json.dump(Features_data, outfile)


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
Matching_data = {}
Matching_data['MatchingPoints'] = []
#
for image_A, image_B in image_Pair:
    matches, image_result = matching.match(image_A, image_B)
    print ("    Matching points beetwin", image_A.id, "and",  image_B.id, "--> Number of Matches (No optimize) ==", len(matches))
    cv.imshow("matches_" + str(image_A.id)+"_"+ str(image_B.id), image_result)

    src_pts = np.float32([ image_A.keyPoints[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    src_desc = [ image_A.des[m.queryIdx] for m in matches ]
    dst_pts = np.float32([ image_B.keyPoints[m.trainIdx].pt for m in matches ]).reshape(-1,2)
    dst_desc = [ image_B.des[m.queryIdx] for m in matches ]


    ###################################################################
    ## ''' Save Matching (Keypoint and descriptor used)Json File ''' ##
    ###################################################################
    features_A = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (src_pts, src_desc)]
    features_B = [{"Px" : json.dumps(str(point[0])), "Py" : json.dumps(str(point[1])), "desc" : json.dumps(desc.tolist(), separators=(',', ':'))} for point, desc in zip (dst_pts, dst_desc)]
   
    Matching_data['MatchingPoints'].append({
        'ID_Matching': "match_" + image_A.id + "_" + image_B.id,
        'matches': [ {"ID_img": image_A.id, "features" : features_A} , {"ID_img": image_B.id, "features" : features_B} ]
    })

    if not path.exists("Json_Files/"):
        os.mkdir("Json_Files/")

    with open('Json_Files/Matching_data.json', 'w', encoding='utf-8') as outfile:
        json.dump(Matching_data, outfile)            
