###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################

from os import listdir
import os.path
from os import path
import open3d as o3d

import itertools

from config import *
from Utils import *
from Save_Load_cameraParams import *

from Image_ import *
from DescriptorMatcherConfig import *
from Matching import *
from HomographyMatEstimation import *
from FundamentalMatEstimation import *
from EssentialMatEstimation import *
from PoseEstimation import *
from P_cloud import *

import copy


class SfM:
    def __init__(self):      
        self.last_images = []
        self.__config()   


    def __config(self):
        ###########################################################
        ##          ''' Define Detector and Matcher '''          ##
        ##             ''' Matching configuration '''            ##
        ##      ''' Config params of Fundamental Matrix '''      ##
        ##       ''' Config params of Essential Matrix '''       ##
        ###########################################################
        self.detector, self.matcher= keyPointDetectorAndMatchingParams(keyPointDetectorMethod = configuration["feature_type"], matcherMethod = configuration["matcher_type"], crosscheck = configuration["matcher_crosscheck"], Max_keyPoint = configuration["max_keyPoint"])      
        assert not self.detector == None and not self.matcher == None, "Problems !!!"
        print ("\nDetector == ", type(self.detector), " Matcher = ", type(self.matcher))

        print ("Matching configuration")
        self.matchingConfig = MatchingConfig(self.matcher, configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])
        # print ("    ", configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])

        print ("Homography Matrix Configuration")
        self.Homographymat = HomographyMatrix(methodOptimizer = configuration["homography_method"])

        print ("Fundamental Matrix Configuration")
        self.Fundamentalmat = FundamentalMatrix(methodOptimizer = configuration["fundamental_method"])

        print ("Essential Matrix Configuration")
        self.Essentialmat = EssentialMatrix(methodOptimizer = configuration["essential_method"])

        ###########################################################
        ##        ''' Load Camera Matrix and update it '''       ##
        ##          ''' Load distortion coefficients '''         ##
        ###########################################################
        self.ImageFolder = "Imgs/Saint_Roch_new/data/"
        # self.ImageFolder = "Imgs/Aguesseau/data/"
        self.cameraParams = load_cameraParams(self.ImageFolder+"camera_Samsung_s7.yaml")

    
    def loadData(self):
        assert path.exists(self.ImageFolder), 'veuillez verifier le chemin du Folder'
        images_name = sorted([file for file in listdir(self.ImageFolder) if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".png")])


        ###########################################################
        ##                  ''' Create Images '''                ##
        ###########################################################
        print ("\nRead images")
        self.remaining_images = []
        for image in images_name:
            self.remaining_images.append(Image_(self.ImageFolder, image, configuration["feature_process_size"], copy.deepcopy(self.cameraParams)))
        print ("    Images = ", images_name)


        ###########################################################
        ##       ''' Calculate KeyPoint and Descriptor '''       ##
        ###########################################################
        print ("\nDetect and compute descriptors")
        for image in self.remaining_images:
            image.keyPointDetector(self.detector)
            print ('\tsize of descriptors of', image.id, image.des.shape)
            

        ###########################################################
        ##  ''' Compute Matching beetwin each pair of image '''  ##
        ###########################################################
        print ("\nMatching points beetwin each pair of image")

        image_Pair = list(itertools.combinations(self.remaining_images, 2))

        self.matches_vector = []
        for image_A, image_B in image_Pair:
            """ Create matches, compute homography_matrix, compute fundamental_matrix, compute essential_matrix """
            matching_AB = Matching(self.matchingConfig, image_A, image_B)
            matching_AB.homographyMatrix_inliers(self.Homographymat)
            
            if matching_AB.homog_mask_len < configuration["homog_threashold"]:
                matching_AB.homog_mask_len = 0

            self.matches_vector.append(matching_AB)
        

    def init(self):
        ###############################################################
        ##  ''' Retrieve best candidate pair for initialization '''  ##
        ###############################################################

        # print  ("Phase Init :")
        # print  ("\tImages disponibles :")
        # [print ("\t\t",x.id) for x in self.remaining_images]
        # print  ("\tImages utilisées :")
        # [print ("\t\t",x.id) for x in self.last_images]
        # print  ("\tmatches disponibles :")
        # [print ("\t\t",x.image_A.id,"  ", x.image_B.id, "   ", x.homog_mask_len ) for x in self.matches_vector]

        index_candidate_pair = np.argmax([x.homog_mask_len for x in self.matches_vector])
        matching_AB = self.matches_vector[index_candidate_pair]
        self.matches_vector.remove(matching_AB)
       
        matching_AB.remove_outliers(self.Fundamentalmat, self.Essentialmat)

        print ("\n-Retrieve best candidate pair for initialization: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")

        ################################################################
        ##           ''' Compute Transformation Matrix '''            ##
        ##      ''' Triangulation and Initialize point-Cloud '''      ##
        ################################################################
        print ("\tCompute transformation and initialize point-Cloud")

        ''' Estimate Pose '''
        cameraPose = PoseEstimation()
        inliers_count, Rotation_Mat, Transl_Vec = cameraPose.FindPoseEstimation_RecoverPose(    matching_AB.essential_matrix,     
                                                                                                matching_AB.curr_pts,             matching_AB.prec_pts, 
                                                                                                matching_AB.curr_pts_norm,        matching_AB.prec_pts_norm,  
                                                                                                matching_AB.image_B.cameraMatrix, matching_AB.image_A.cameraMatrix,
                                                                                                matching_AB.inliers_mask)   # updated with referece argument
        print ("\tEstimate-pose Recover_pose : {} inliers / {} --->".format(inliers_count, len(matching_AB.inliers_mask)))

        ''' set absolute Pose '''
        matching_AB.image_A.setAbsolutePose(None, np.eye(3), np.zeros(3))
        matching_AB.image_B.setAbsolutePose(matching_AB.image_A, Rotation_Mat, Transl_Vec)

        ''' Generate 3D_points using Triangulation ''' 
        matching_AB.generate_landmarks()
       
        self.last_images.append(matching_AB.image_A)
        self.last_images.append(matching_AB.image_B)

        self.remaining_images.remove(matching_AB.image_A)
        self.remaining_images.remove(matching_AB.image_B)


    def incremental(self):
        while(len(self.remaining_images) > 0):
            matching_vector = self.__best_new_image()
           
            ################################################################
            ##       ''' Compute Transformation Matrix using PNP '''      ##
            ##       ''' Triangulation and increment point-Cloud '''      ##
            ################################################################

            if not(matching_vector == None):
                
                if len(matching_vector) == 1:
                    matching_AB = matching_vector[0]
                    matching_AB.remove_outliers(self.Fundamentalmat, self.Essentialmat)
                    if configuration["incremental_method_is_resection"] == False:
                        matching_AB.computePose_2D2D_Scale(self.Essentialmat)
                    else:

                        print ("\tRetrieve 3D_points betwwen pair matches to pnp")
                        # matching_AB.remove_outliers(self.Fundamentalmat, self.Essentialmat)
                        matching_AB.update_inliers_mask()
                        Result= self.__points_to_pnp_singal(matching_AB)
                        if Result is None:
                            ## Boucle à l'infinie dans ce cas là, il faut trouver une solution
                            continue
                    
                        inter_3d_pts, inter_2d_pts, inter_2d_pts_norm = Result

                        print ("\tresection using pnp")
                        Result = self.__estimateCameraPose_pnp(inter_3d_pts, inter_2d_pts, inter_2d_pts_norm, matching_AB.image_B.cameraMatrix)
                        if Result is None: 
                            # pas interesant la nouvelle image
                            continue
                        inliers_Number, Rot, transl, mask_inliers = Result

                        """ Remove outliers of PnP Ransac """
                        inter_3d_pts      = np.asarray(inter_3d_pts)     [mask_inliers.ravel() > 0].reshape(-1, 3)
                        inter_2d_pts      = np.asarray(inter_2d_pts)     [mask_inliers.ravel() > 0].reshape(-1, 2)
                        inter_2d_pts_norm = np.asarray(inter_2d_pts_norm)[mask_inliers.ravel() > 0].reshape(-1, 2)

                        """ Add inlier informations of PnP Ransac to the newest image """
                        matching_AB.image_B.points_3D_used       = np.append(matching_AB.image_B.points_3D_used,      inter_3d_pts,      axis = 0)
                        matching_AB.image_B.points_2D_used       = np.append(matching_AB.image_B.points_2D_used,      inter_2d_pts,      axis = 0)
                        matching_AB.image_B.points_2D_norm_used  = np.append(matching_AB.image_B.points_2D_norm_used, inter_2d_pts_norm, axis = 0)
        
                        """Set camera pose of new image"""
                        matching_AB.image_B.setAbsolutePose(None, Rot, transl)
                        
                        print("\tTriangulate with all images")
                        matching_AB.generate_landmarks()

                        self.last_images.append(matching_AB.image_B)
                        self.remaining_images.remove(matching_AB.image_B)


                else:
                    print ("\tRetrieve 3D_points from all pair matches to pnp")
                    [matching_AB.remove_outliers(self.Fundamentalmat, self.Essentialmat)  for matching_AB in matching_vector]
                    [matching_AB.update_inliers_mask()  for matching_AB in matching_vector]

                    inter_3d_pts, inter_2d_pts, inter_2d_pts_norm, mask = self.__points_to_pnp_multiple(matching_vector)

                    print ("\tresection using pnp")
                    Result = self.__estimateCameraPose_pnp(inter_3d_pts, inter_2d_pts, inter_2d_pts_norm, matching_vector[0].image_B.cameraMatrix)
                    if Result is None: 
                        # pas interesant la nouvelle image
                        return
                    inliers_Number, Rot, transl, mask_inliers = Result

                    """ Remove outliers of PnP Ransac """
                    inter_3d_pts      = np.asarray(inter_3d_pts)     [mask_inliers.ravel() > 0].reshape(-1, 3)
                    inter_2d_pts      = np.asarray(inter_2d_pts)     [mask_inliers.ravel() > 0].reshape(-1, 2)
                    inter_2d_pts_norm = np.asarray(inter_2d_pts_norm)[mask_inliers.ravel() > 0].reshape(-1, 2)

                    """ Add inlier informations of PnP Ransac to the newest image """
                    matching_vector[0].image_B.points_3D_used       = np.append(matching_vector[0].image_B.points_3D_used,      inter_3d_pts,      axis = 0)
                    matching_vector[0].image_B.points_2D_used       = np.append(matching_vector[0].image_B.points_2D_used,      inter_2d_pts,      axis = 0)
                    matching_vector[0].image_B.points_2D_norm_used  = np.append(matching_vector[0].image_B.points_2D_norm_used, inter_2d_pts_norm, axis = 0)
                    
                    """Set camera pose of new image"""
                    matching_vector[0].image_B.setAbsolutePose(None, Rot, transl)

                    print("\tTriangulate with all images")
                    [matching_AB.generate_landmarks() for index_matching, matching_AB in enumerate(matching_vector) if mask[index_matching]]

                    self.last_images.append(matching_vector[0].image_B)
                    self.remaining_images.remove(matching_vector[0].image_B)


    def __points_to_pnp_singal(self, matching_AB):
        ''' Intersection 3D 2D '''
        Result = matching_AB.retrieve_existing_points()
        """ pas assez de points """
        if Result is None: 
            return  

        print("\tNumber of 3D_Point to resection is {} points".format(len(Result[0])))
        matching_AB.points_for_triangulate()

        return Result



    def __points_to_pnp_multiple(self, matching_vector):
        pts_cloud_for_pnp     = np.empty([0, 3])
        pts_curr_for_pnp      = np.empty([0, 2])
        ptd_curr_norm_for_pnp = np.empty([0, 2])
        mask                  = np.ones(len(matching_vector))
        for index_match, matching_AB in enumerate(matching_vector):
            ''' Intersection 3D 2D '''
            Result = matching_AB.retrieve_existing_points()
            """ pas assez de points """
            if Result is None: 
                """ remove it : pour ne aps le prendre en cosideration lors de la triangulation """
                # matching_vector.remove(matching_AB)
                mask[index_match] = 0
                continue 
            inter_3d_pts, inter_2d_pts, inter_2d_pts_norm = Result
            
            matching_AB.points_for_triangulate()

            for idx in range (len(inter_3d_pts)):
                if (inter_3d_pts[idx] in pts_cloud_for_pnp and inter_2d_pts[idx] in pts_curr_for_pnp and inter_2d_pts_norm[idx] in ptd_curr_norm_for_pnp ):
                    continue
                else:
                    pts_cloud_for_pnp     = np.append(pts_cloud_for_pnp     , [inter_3d_pts[idx]]     , axis = 0)
                    pts_curr_for_pnp      = np.append(pts_curr_for_pnp      , [inter_2d_pts[idx]]     , axis = 0)
                    ptd_curr_norm_for_pnp = np.append(ptd_curr_norm_for_pnp , [inter_2d_pts_norm[idx]], axis = 0)
            
        print("\tNumber of 3D_Point to resection is {} points".format(len(pts_cloud_for_pnp)))
        
        return pts_cloud_for_pnp, pts_curr_for_pnp, ptd_curr_norm_for_pnp, mask


    
    def __estimateCameraPose_pnp(self, pts_cloud_for_pnp, pts_curr_for_pnp, ptd_curr_norm_for_pnp, cameraMatrix):  
        pose = PoseEstimation()
        return pose.FindPoseEstimation_pnp( pts_cloud_for_pnp, pts_curr_for_pnp, ptd_curr_norm_for_pnp, cameraMatrix)
        print("\t\tnumber of 3D points to project on the new image for estimate points is {} points".format(len(pts_cloud_for_pnp)))


    # /****************************************************************/
    # /*****     Function of best image for inceremntal phase     *****/
    # /*****         using number of homography criterion         *****/          
    # /****************************************************************/
    def __best_new_image(self):

        if configuration["pnpsolver_method"] and configuration["use_allimages"]:

            print("\n\t/************************** Choose best new image *****************************/")
            # /** ^ == XOR **/
            """ Recuperer la meilleures images qui partagent le plus de matches ( avec le mask d'homography) avec l'une des images qui a été deja été utilisée """
            matches_vector = [x for x in self.matches_vector if ((x.image_A in self.last_images) ^ (x.image_B in self.last_images))]
            my_dict = {}
            for new_image in self.remaining_images:
                nb_matches_homog = []
                matches_vector_this_image = [x for x in matches_vector if ((x.image_A.id == new_image.id) ^ (x.image_B.id == new_image.id)) and x.homog_mask_len > 0]
                [nb_matches_homog.append(x.homog_mask_len) for x in matches_vector_this_image]
                # TODO: add condition "threshold"
                nb_matches_homog = np.sum(nb_matches_homog)
                my_dict[new_image] = nb_matches_homog
                print("\timage {} --> number total matches homography {}".format(new_image.id, nb_matches_homog))
            print("\t/*****************************************************************************/")

            my_new_image = max(my_dict, key = my_dict.get)
            # if my_dict[my_new_image] < 100:
            #     matches_vector = [x for x in matches_vector if ((x.image_A.id == my_new_image.id) ^ (x.image_B.id == my_new_image.id))  and x. > 0]


            print ("\n-Retrieve best candidate image for incremetal phase is ",my_new_image.id)  
           
            """ out matching for pnp and trianglutae """
            matches_vector = [x for x in matches_vector if ((x.image_A.id == my_new_image.id) ^ (x.image_B.id == my_new_image.id))]

            for matching_AB in matches_vector:
                if(matching_AB.image_B in self.last_images and matching_AB.image_A not in self.last_images):
                    matching_AB.permute_prec_curr()

            return matches_vector

        if configuration["pnpsolver_method"] and not configuration["use_allimages"]:
            # /***************************************************************/
            # /** ''' Retrieve best candidate pair for incremetal phase ''' **/
            # /***************************************************************/
            """ Recuprer la pair d'image où une des deux images à déja été utilisée """
            print("\timage {} --> number total matches homography {}".format(new_image.id, nb_matches_homog))
            [ self.matches_vector.remove(x) for x in (self.matches_vector) if ((x.image_A in self.last_images) and (x.image_B in self.last_images)) ]

            if len(self.matches_vector) == 0:
                # il faut améliorer l'algo
                return None
                
            # /** ^ == XOR **/
            """ Recuperer la meilleures images qui partagent le plus de matches ( avec l'homography ) avec l'une des images qui a été deja été utilisée """
            matches_vector = [x for x in self.matches_vector if ((x.image_A in self.last_images) ^ (x.image_B in self.last_images))]

            if len(matches_vector) == 0:
                print("\nles deux images on déjà été traité !!!")
                return None
           
            index_candidate_pair = np.argmax([x.homog_mask_len for x in matches_vector])
            matching_AB = matches_vector[index_candidate_pair]
            self.matches_vector.remove(matching_AB)
            
            # Rectify precedent_image, current_image
            if(matching_AB.image_B in self.last_images and matching_AB.image_A not in self.last_images):
                matching_AB.permute_prec_curr()

            print ("\n-Retrieve best candidate pair for incremetal phase: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")  

            return [matching_AB]

            
        if not configuration["pnpsolver_method"]:
            print("Not implemented yet")
            return None


    def save_data_for_BA(self):
        import time
        start = time.time()
        pp_cloud = P_cloud(self.last_images)
        done = time.time()
        elapsed = done - start
        print("elapsed time ... {}".format(elapsed))


    # /****************************************************************/
    # /*****    Function of display Point-Cloud and Camera pos    *****/          
    # /****************************************************************/
    def display_reconstruction(self):
        """ retrieve all 3D-points to draw """
        p_cloud_list = [img.points_3D_used for img in self.last_images]
        p_cloud = union(p_cloud_list)

        print ("\n- Number of 3D points of 3D-reconstruction generated with SfM is  ",len(p_cloud), "points")

        """  Save point-cloud in ply file  """
        pts2ply(p_cloud, filename = "src/point_cloud.ply")
        
        """ Config viewer """
        vis = o3d.visualization.Visualizer()
        vis.create_window("Structure_from_Motion", 1280, 720)

        """  Add Camera-pose of images to viewer """
        scale = 0.5
        for img in (self.last_images):
            if("transform" in img.absoluteTransformation.keys()):

                mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scale).transform(img.absoluteTransformation["transform"])
                id_ref_imd_id = text_3d(img.id, img.absoluteTransformation["transform"][0:3, 3], degree = 180.)
                
                vis.add_geometry(mesh_img_i)
                vis.add_geometry(id_ref_imd_id)

        """  Add point-Cloud of 3D-reconstruction to viewer """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(p_cloud).reshape(-1, 3))

        vis.add_geometry(pcd)

        """  Launch Visualisation  """
        vis.run()



    ##TODO 2D2D_Scale
    """
        def computePose_2D2D_Scale(self, Fundamentalmat, Essentialmat):
            self.remove_outliers(Fundamentalmat, Essentialmat)

            ''' Estimate Pose '''
            inliers_count, Rotation_Mat, Transl_Vec, self.inliers_mask = PoseEstimation().EstimatePose_from_2D2D_scale(self, Essentialmat)
            print ("\tEstimate-pose 2D-2D: ", inliers_count,"inliers /",len(self.prec_pts))
            

            ###########################################################
            ##    ''' Generate 3D_points using Triangulation '''     ##
            ###########################################################
            print ("\tGenerate 3D_points using Triangulation")
            ''' Remove all outliers and triangulate'''
            self.update_inliers_mask()   # Take just inliers for triangulation
            points3d, index_to_Remove = Triangulation().Triangulate(self)
            
            # Update inliers points and descriptors
            p_cloud            = np.delete(points3d         , index_to_Remove, axis=0)

            prec_pts_inliers_F  = np.delete(self.prec_pts     , index_to_Remove, axis=0)
            prec_pts_norm_      = np.delete(self.prec_pts_norm, index_to_Remove, axis=0)
            # prec_desc_inliers_F = np.delete(self.prec_desc    , index_to_Remove, axis=0)

            curr_pts_inliers_F  = np.delete(self.curr_pts     , index_to_Remove, axis=0)
            curr_pts_norm_F     = np.delete(self.curr_pts_norm, index_to_Remove, axis=0)
            # curr_desc_inliers_F = np.delete(self.curr_desc    , index_to_Remove, axis=0)

            
            _, index_last, index_new = intersect2D(self.image_A.points_2D_used, prec_pts_inliers_F)
            
            p_cloud_new = p_cloud[index_new]
            p_cloud_old = self.image_A.points_3D_used[index_last]
            
            #Thibaud Method
            scale_ratio_last = [np.linalg.norm(p_cloud_old[i] - p_cloud_old[j]) for i in range(0, len(p_cloud_old) - 1) for j in range(i+1, len(p_cloud_old))]
            scale_ratio_new  = [np.linalg.norm(p_cloud_new [i] - p_cloud_new [j]) for i in range(0, len(p_cloud_new)  - 1) for j in range(i+1, len(p_cloud_new ))]
            scale = [(x / y) for x, y in zip(scale_ratio_last , scale_ratio_new) ]
            print (np.array(scale).reshape(-1))

            print (Transl_Vec)
            print (np.mean(scale))
            Transl_Vec = Transl_Vec * np.mean(scale)#([xScale, yScale, zScale]).reshape(-1, 1)  
            print (Transl_Vec)
            ''' set absolute Pose '''
            self.image_B.setAbsolutePose(self.image_A, Rotation_Mat, Transl_Vec)
            self.image_B.absoluteTransformation["transform"] = self.image_A.absoluteTransformation["transform"] @ self.image_B.absoluteTransformation["transform"]
            self.image_B.absoluteTransformation["projection"] = projection_from_transformation(self.image_B.absoluteTransformation["transform"])


    """