import os
import yaml

default_config_yaml = '''
# Params for features
feature_type: ORB                       # Feature type (SURF, SIFT, ORB, AKAZE)
max_keyPoint: 20000
feature_process_size: 1024              # Resize the image if its size is larger than specified. Set to -1 for original size

# Params for general matching
lowes_ratio: 0.75                        # Ratio test for matches
matcher_type: BFMatcher                # FlannMatcher, BFMatcher
matcher_crosscheck: True                 # True (match) or False (knnmatch)
symmetric_matching: True                # Match symmetricly or one-way
symmetric_matching_type: intersection   # intersection or union

# Params Fundamental_Matrix --> >= 5 points
fundamental_method: 8                       # 1(FM_7POINT)[== 7 points], 2(FM_8POINT)[>= 8 points], 4(cv.FM_LMEDS)[>= 8 points], 8(cv.FM_RANSAC)[>= 8 points]

# Params Essential_Matrix --> >= 5 points
essential_method: 8                       # 4(cv.LMEDS), 8(cv.RANSAC)

# Params Resection 
# pose_estimation
undistort_point: True

# Params Resection 
pnpsolver_method: 1                       # 0(solvepnp), 1(solvepnpRansac)
'''


def default_config():
    """Return default configuration"""
    return yaml.safe_load(default_config_yaml)

###########################################################
##            ''' Load Configuration File '''            ##
###########################################################
configuration = default_config()
