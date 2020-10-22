import cv2 as cv


class FundamentalMatrix:
    def __init__(self, methodOptimizer = cv.FM_RANSAC, ransacReprojThreshold = 3.0):
        self.methodOptimizer = methodOptimizer
        self.ransacReprojThreshold = ransacReprojThreshold


    def compute_FundamentalMatrix (self, matching):
        NB_Matching_Threshold = 8
        if len(matching.prec_pts) < NB_Matching_Threshold:
            return None, None
        self.FundamentalMat, maskInliers = cv.findFundamentalMat(matching.curr_pts, matching.prec_pts, method = self.methodOptimizer, ransacReprojThreshold = self.ransacReprojThreshold)

        return maskInliers.reshape(-1)
            