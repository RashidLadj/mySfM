from OpenCV_version import *
import cv2

'''
  /** This Method Allows to configure the system by choosing the method of detection of points of interest (Sift, Surf, ORB), 
  *  as well as the Matching method (FlannBasedMatcher, BruteForceMatcher).
  *  Ps: In the case of CrossCheck == True, only Matcher.match Works 
  **/

  /** Cette Methode Permet de configurer le system en choisissant la methode de detection de points d'interet (Sift, Surf, ORB), 
  *  ainsi que la methode de Matching (FlannBasedMatcher, BruteForceMatcher)
  *  Ps: Dans le cas du CrossCheck == True, seul Matcher.match Fonctionne 
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
        #   assert 1==0 , "Introduire SURF de OpenCV2"

        if is_CV3():
            detector = cv.features2d.SURF_create()
            # detector = cv.SURF_create()

        if is_CV4():
            detector = None
            assert 1==0 , "SURF ne marche pas sur OpenCV4"

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
        #   assert 1==0 , "Introduire Akaze de OpenCV2"

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
        
        matcher = cv.FlannBasedMatcher(flann_params, dict(checks = 32))  # bug : need to pass empty dict (#1329)

    else:
        matcher = cv.BFMatcher(normType = norm, crossCheck = crosscheck)
    
    return detector, matcher    


