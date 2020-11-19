import cv2 as cv


class FundamentalMatrix:
    def __init__(self, methodOptimizer = cv.FM_RANSAC, ransacReprojThreshold = 3.0):
        self.methodOptimizer = methodOptimizer
        self.ransacReprojThreshold = ransacReprojThreshold


    def compute_FundamentalMatrix (self, src_pts, dst_pts, image_A, image_B, matches):

        # minVal, maxVal, _, _ = cv.minMaxLoc(dst_pts)
        # self.ransacReprojThreshold = 0.006 * maxVal

        NB_Matching_Threshold = 8
        if len(src_pts) < NB_Matching_Threshold:
            return None
        _, maskInliers = cv.findFundamentalMat(src_pts, dst_pts, method = self.methodOptimizer, ransacReprojThreshold = self.ransacReprojThreshold)

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                 singlePointColor = None,
        #                 matchesMask = maskInliers, # draw only inliers
        #                 flags = 2)

        # goodMatchesImage = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints, matches, None, **draw_params)
        # cv.imshow("Fundamental_"+str(image_A.id)+"_"+str(image_B.id), goodMatchesImage)

        return maskInliers.reshape(-1)
            
            