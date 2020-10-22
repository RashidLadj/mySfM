import os
import yaml

default_config_yaml = '''
# Params for features
feature_type: ORB                       # Feature type (SURF, SIFT, ORB, AKAZE)
max_keyPoint: 200000
feature_process_size: 1024              # Resize the image if its size is larger than specified. Set to -1 for original size

# Params for general matching
lowes_ratio: 0.75                        # Ratio test for matches
matcher_type: BFMatcher                # FlannMatcher, BFMatcher
matcher_crosscheck: True                 # True (match) or False (knnmatch)
symmetric_matching: False                # Match symmetricly or one-way
symmetric_matching_type: intersection   # intersection or union

# Params Homography_Matrix --> >= 4 points
enable_homographyMat : True
homography_method: 8                       ## /** 0(None), 4(cv.LMEDS), 8(cv.RANSAC), 16(RHO) **/ ##

# Params Fundamental_Matrix --> >= 5 points
enable_fundamentalMat : True
fundamental_method: 8                       # 1(FM_7POINT)[== 7 points], 2(FM_8POINT)[>= 8 points], 4(cv.FM_LMEDS)[>= 8 points], 8(cv.FM_RANSAC)[>= 8 points]

# Params Essential_Matrix --> >= 5 points
essential_method: 8                       # 4(cv.LMEDS), 8(cv.RANSAC)

# pose_estimation
undistort_point: True

# params incremental sfm
incremental_method_is_resection: 1                   #0 ( 2D-2D method -> recoverose + scale), 1 (Resection usning pnp)

# Params Resection 
pnpsolver_method: 1                       # 0(solvepnp), 1(solvepnpRansac)
use_allimages: False                       # if pnp is used, choose if you cant use just last image, or all last image for intersection 3D points
'''


def default_config():
    """Return default configuration"""
    return yaml.safe_load(default_config_yaml)
 ##
configuration = default_config()
