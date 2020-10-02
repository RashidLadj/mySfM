import cv2 as cv
import numpy as np

class Matching:
    def __init__(self, Matcher, Symmetric, Symmetric_Type, CrossCheck, Lowes_ratio):
        self.matcher = Matcher                  # FlannMatcher or BFmatcher
        self.symmetric = Symmetric              # True or False
        self.symmetric_Type = Symmetric_Type    # Intersection or union
        self.crosscheck = CrossCheck            # true or false
        self.lowes_ratio = Lowes_ratio          # Lowes_ratio for knnMatch
        self.matchMethod = self.__config_method()   # method to use [MatchingSimple, MatchingIntersection , MatchingUnion]                       


    def __config_method(self):
        if not self.symmetric:
            return self.__MatchingSimple
        elif self.symmetric and self.symmetric_Type == "intersection" :
            return self.__MatchingIntersection
        elif self.symmetric and self.symmetric_Type == "union" :
            return self.__MatchingUnion
        else:
            print ("Configuration False")
            return None


    def match(self, image_A, image_B) :
        matches, Result = self.matchMethod(image_A, image_B)

        return matches, Result    


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


    def __permuteValue(self, item):
        temp = item.trainIdx
        item.trainIdx = item.queryIdx
        item.queryIdx = temp


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
        [self.__permuteValue(item) for item in matches_BA_ConvertToAB]

        intersectionMatches = []
        for itemAB in matches_AB:
            for itemBA in matches_BA_ConvertToAB:
                if self.__areEqual(itemAB, itemBA) :   # Je n'ai pas trouvé un autre moyen de faire la comparaison
                    intersectionMatches.append(itemAB)

        # Sort them in the order of their distance.
        intersectionMatches = sorted(intersectionMatches, key = lambda x: x.distance)   

        # take just 2/3 ?? (A revoir)
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
        [self.__permuteValue(item) for item in matches_BA_ConvertToAB]

        unionMatches = []
        [unionMatches.append(item) for item in matches_AB]
        for itemBA in matches_BA_ConvertToAB:
            exist = False
        for item in unionMatches:
            if self.__areEqual(item, itemBA) :
                exist = True
            break   # Je n'ai pas trouvé un autre moyen de faire la comparaison
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