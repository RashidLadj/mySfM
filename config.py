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
matcher_crosscheck: False                 # True (match) or False (knnmatch)
symmetric_matching: True                # Match symmetricly or one-way
symmetric_matching_type: intersection   # intersection or union

'''


def default_config():
    """Return default configuration"""
    return yaml.safe_load(default_config_yaml)
