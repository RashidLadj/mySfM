from OpenCV_version import *
import cv2

'''
  /** This Method Allows to configure the system by choosing the method of detection of points of interest (Sift, Surf, ORB), 
  *  as well as the Matching method (FlannBasedMatcher, BruteForceMatcher).
  *  Ps: In the case of CrossCheck == True, only Matcher.match Works 
  **/
'''

def keyPointDetectorAndMatchingParams(keyPointDetectorMethod, matcherMethod, crosscheck = False, Max_keyPoint = 500):
    if keyPointDetectorMethod.lower() == 'sift':
        # if is_CV2():
        #   assert 1==0 , "Introduire SIFT de OpenCV2"

        # for opencv 3.x and above
        if is_CV3():
            # > 3.4.2.17  don't work 
            detector = cv2.xfeatures2d.SIFT_create()

        if is_CV4(): 
            detector = cv.SIFT_create() 
            # or
            # detector = cv.SIFT().create() 

        norm = cv.NORM_L2

    elif keyPointDetectorMethod.lower() == 'surf':
        # if is_CV2():
        #   assert 1==0 , "Add SURF from OpenCV2"

        if is_CV3():
            detector = cv.features2d.SURF_create()
            # detector = cv.SURF_create()

        if is_CV4():
            detector = None
            assert 1==0 , "SURF does not work in OpenCV4"

        norm = cv.NORM_L2

    elif keyPointDetectorMethod.lower() == 'orb':
        if is_CV3() or is_CV4():
          detector = cv.ORB_create( nfeatures = Max_keyPoint )
          
        if is_CV2():
          detector = cv2.ORB( nfeatures = Max_keyPoint )

        norm = cv.NORM_HAMMING

    elif keyPointDetectorMethod.lower() == 'akaze':
        if is_CV3() or is_CV4():
          detector = cv.AKAZE_create()
          
        # if is_CV2():
        #   assert 1==0 , "Add Akaze from OpenCV2"

        norm = cv.NORM_HAMMING

    else:
        return None, None

    if matcherMethod.lower() == "flannmatcher":
        #https://github.com/opencv/opencv/blob/383559c2285baaf3df8cf0088072d104451a30ce/modules/flann/include/opencv2/flann/defines.h#L68
        if norm == cv.NORM_L2:
            FLANN_INDEX_KDTREE = 1  # For SIFT, SIRF ...
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

        else:
            FLANN_INDEX_LSH = 6      # For ORB, BRIEF ...
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6,      # 12
                               key_size = 12,         # 20
                               multi_probe_level = 1) #2
        
        matcher = cv.FlannBasedMatcher(flann_params, dict(checks = 32))  # bug: need to pass empty dict (#1329)

    else:
        matcher = cv.BFMatcher(normType = norm, crossCheck = crosscheck)
    
    return detector, matcher    


""" Just Configuration Class """
class MatchingConfig:
    def __init__(self, Matcher, Symmetric, Symmetric_Type, CrossCheck, Lowes_ratio):
        self.matcher = Matcher                  # FlannMatcher or BFmatcher
        self.symmetric = Symmetric              # True or False
        self.symmetric_Type = Symmetric_Type    # Intersection or union
        self.crosscheck = CrossCheck            # true or false
        self.lowes_ratio = Lowes_ratio          # Lowes_ratio for knnMatch
        self.matchMethod = self.__config_method()   # method to use [MatchingSimple, MatchingIntersection , MatchingUnion]      


    def __config_method(self):
        if not self.symmetric:
            print("- matching assymetric")
            return self.__MatchingSimple
        elif self.symmetric and self.symmetric_Type == "intersection" :
            print("- matching symetric intersection")
            return self.__MatchingIntersection
        elif self.symmetric and self.symmetric_Type == "union" :
            print("- matching symetric union")
            return self.__MatchingUnion
        else:
            print ("Configuration False")
            return None


    ###################################################################
    ##       ''' Retrieve matches from A to B (Assymetric) '''       ##
    ###################################################################
    def __MatchingSimple(self, image_A, image_B):
        if self.crosscheck:
            # Methode One
            matches = self.matcher.match(image_A.des, image_B.des)
            # take just 2/3 ?? (A revoir)
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw All matches.
            image_matches = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        else:
            # Methode Two
            matches = self.matcher.knnMatch(image_A.des, image_B.des, k=2)
            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < self.lowes_ratio * n.distance:
                    good.append([m])
                    #matches.append(m)
            matches = [item[0] for item in good]

            # Draw All matches.
            image_matches = cv.drawMatchesKnn(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,good, None, flags=2)
        
        # Sort them in the order of their distance.
        # matches = sorted(matches, key = lambda x:x.distance)

        return matches, image_matches    


    def __areEqual(self, itemAB, itemBA):
        return itemAB.trainIdx == itemBA.trainIdx and itemAB.queryIdx == itemBA.queryIdx and itemAB.distance == itemBA.distance and itemAB.imgIdx == itemBA.imgIdx


    ####################################################################
    ##   ''' Retrieve matches Symmetricly and take intersection '''   ##
    ####################################################################
    def __MatchingIntersection(self, image_A, image_B):
        if self.crosscheck:
            # Methode One
            ''' Match Img A to img B '''
            matches_AB = self.matcher.match(image_A.des, image_B.des)
            
            ''' Match Img B to img A '''
            matches_BA = self.matcher.match(image_B.des, image_A.des)

        else:
            # Methode Two
            ''' Match Img A to img B '''
            matches_AB = self.matcher.knnMatch(image_A.des, image_B.des, k=2)
            
            # Apply ratio test
            good_AB = []
            for m, n in matches_AB:
                if m.distance < self.lowes_ratio * n.distance:
                    good_AB.append([m])  # If need to use drawKnnMatch
                    #matches_ABF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_AB = [(item[0]) for item in good_AB]

            ''' Match Img B to img A '''
            matches_BA = self.matcher.knnMatch(image_B.des, image_A.des, k=2)

            # Apply ratio test
            good_BA = []
            for m, n in matches_BA:
                if m.distance < self.lowes_ratio * n.distance:
                    good_BA.append([m])  # If need to use drawKnnMatch
                    #matches_BAF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_BA = [(item[0]) for item in good_BA]

        matches_BA_ConvertToAB = matches_BA.copy()
        
        def permuteValue(item):
            item.trainIdx, item.queryIdx = item.queryIdx, item.trainIdx
        [permuteValue(item) for item in matches_BA_ConvertToAB]

        intersectionMatches = []
        for itemAB in matches_AB:
            for itemBA in matches_BA_ConvertToAB:
                if self.__areEqual(itemAB, itemBA) :   # Je n'ai pas trouvé un autre moyen de faire la comparaison
                    intersectionMatches.append(itemAB)

        # Sort them in the order of their distance.
        intersectionMatches = sorted(intersectionMatches, key = lambda x: x.distance)   

        # take just 2/3 ?? (TODO)
        # matches = sorted(matches, key = lambda x:x.distance)

        goodInter = intersectionMatches.copy()
        goodInterKnn = [[item] for item in intersectionMatches]

        # Matching 
        good_Without_Optimizer = len(intersectionMatches)
        image_matches = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodInter, None, flags=2) if self.crosscheck else cv.drawMatchesKnn(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodInterKnn, None, flags=2)

        return intersectionMatches, image_matches    


    ####################################################################
    ##       ''' Retrieve matches Symmetricly and take union '''      ##
    ####################################################################
    def __MatchingUnion(self, image_A, image_B):
        if self.crosscheck:
                # Method One
            ''' Match Img A to img B '''
            matches_AB = self.matcher.match(image_A.des, image_B.des)
            
            ''' Match Img B to img A '''
            matches_BA = self.matcher.match(image_B.des, image_A.des)

        else:
            # Method Two
            ''' Match Img A to img B '''
            matches_AB = self.matcher.knnMatch(image_A.des, image_B.des, k=2)

            # Apply ratio test
            good_AB = []
            for m,n in matches_AB:
                if m.distance < self.lowes_ratio * n.distance:
                    good_AB.append([m])  # If need to use drawKnnMatch
                    #matches_ABF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_AB = [(item[0]) for item in good_AB]

            ''' Match Img B to img A '''
            matches_BA = self.matcher.knnMatch(image_B.des, image_A.des, k=2)
            # Apply ratio test
            good_BA = []
            for m,n in matches_BA:
                if m.distance < self.lowes_ratio * n.distance:
                    good_BA.append([m])  # If need to use drawKnnMatch
                    #matches_BAF.append(m)  # If need to use drawMatch ( Next Step With optimizer )
            matches_BA = [(item[0]) for item in good_BA]

        matches_BA_ConvertToAB = matches_BA.copy()
        def permuteValue(item):
            item.trainIdx, item.queryIdx = item.queryIdx, item.trainIdx
        [permuteValue(item) for item in matches_BA_ConvertToAB]

        unionMatches = []
        [unionMatches.append(item) for item in matches_AB]
        for itemBA in matches_BA_ConvertToAB:
            exist = False
            for item in unionMatches:
                if self.__areEqual(item, itemBA) :
                    exist = True
                break   # I did not find another method to compare
            if not exist:
                unionMatches.append(itemBA)

        # Sort them in the order of their distance.
        unionMatches = sorted(unionMatches, key = lambda x: x.distance)

        goodUnion = unionMatches.copy()
        goodUnionKnn = [[item] for item in unionMatches]

        # Matching 
        good_Without_Optimizer = len(unionMatches)
        image_matches = cv.drawMatches(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodUnion, None, flags=2) if self.crosscheck else cv.drawMatchesKnn(image_A.imageRGB, image_A.keyPoints, image_B.imageRGB, image_B.keyPoints,goodUnionKnn, None, flags=2)

        return unionMatches, image_matches 

