import cv2 as cv


class FundamentalMatrix:
    def __init__(self, methodOptimizer = cv.FM_RANSAC, ransacReprojThreshold = 3.0):
        self.methodOptimizer = methodOptimizer
        self.ransacReprojThreshold = ransacReprojThreshold


    def compute_FondamentalMatrix (self, src_pts, dst_pts):
        NB_Matching_Threshold = 8
        if len(src_pts) < NB_Matching_Threshold:
            return [], []
        
        FondamentalMat, maskInliers = cv.findFundamentalMat(src_pts, dst_pts, method = self.methodOptimizer, ransacReprojThreshold = self.ransacReprojThreshold)
        return FondamentalMat, maskInliers
            

    