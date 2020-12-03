###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################

import os
from os import path
import open3d as o3d

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

import copy


class Reloc:
    def __init__(self, last_images):      
        self.last_images = last_images
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
        self.matchingConfig = MatchingConfig(self.matcher, configuration["symmetric_matching"], configuration["symmetric_matching_type"], configuration["matcher_crosscheck"], configuration["lowes_ratio"])
        
        self.Homographymat = HomographyMatrix(methodOptimizer = configuration["homography_method"])

        self.Fundamentalmat = FundamentalMatrix(methodOptimizer = configuration["fundamental_method"])

        self.Essentialmat = EssentialMatrix(methodOptimizer = configuration["essential_method"])

        ###########################################################
        ##        ''' Load Camera Matrix and update it '''       ##
        ##          ''' Load distortion coefficients '''         ##
        ###########################################################
        self.ImageFolder = "img/Saint_Roch_new/Reloc/"
        self.cameraParams = load_cameraParams(self.ImageFolder+"camera_Samsung_s7.yaml")

    
    def loadData(self):
        assert path.exists(self.ImageFolder), 'veuillez verifier le chemin du Folder'
        images_name = sorted([file for file in os.listdir(self.ImageFolder) if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".png")])

        ###########################################################
        ##                  ''' Create Images '''                ##
        ###########################################################
        print ("\nRead images (Reloc)")
        self.remaining_images = []
        for image in images_name:
            self.remaining_images.append(Image_(self.ImageFolder, image, configuration["feature_process_size"], copy.deepcopy(self.cameraParams)))
        print ("    Images = ", images_name)


        ###########################################################
        ##       ''' Calculate KeyPoint and Descriptor '''       ##
        ###########################################################
        for image in self.remaining_images:
            image.keyPointDetector(self.detector)
            print ('\tsize of descriptors of', image.id, image.des.shape)
            

        ###########################################################
        ##  ''' Compute Matching between each pair of image '''  ##
        ###########################################################
        print ("\nMatching points between each pair of image")

        for image_A in self.remaining_images:
            matches_vector = []
            for image_B in self.last_images:
                """ Create matches, compute homography_matrix, compute fundamental_matrix, compute essential_matrix """
                matching_AB = Matching(self.matchingConfig, image_B, image_A)
                matching_AB.homographyMatrix_inliers(self.Homographymat)
                
                if matching_AB.homog_mask_len < configuration["homog_threashold"]:
                    matching_AB.homog_mask_len = 0

                matches_vector.append(matching_AB)
            self.__computePose(matches_vector)
        

    def __computePose(self, matches_vector):
        ################################################################
        ##       ''' Compute Transformation Matrix using PNP '''      ##
        ##       ''' Triangulation and increment point-cloud '''      ##
        ################################################################

        matching_vector = self.__best_match(matches_vector)
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
                        return False
                
                    inter_3d_pts, inter_2d_pts, inter_2d_pts_norm = Result

                    print ("\tresection using pnp")
                    Result = self.__estimateCameraPose_pnp(inter_3d_pts, inter_2d_pts, inter_2d_pts_norm, matching_AB.image_B.cameraMatrix)
                    if Result is None: 
                        # pas interesant la nouvelle image
                        return False
                    inliers_Number, Rot, transl, mask_inliers = Result

                    """Set camera pose of new image"""
                    matching_AB.image_B.setAbsolutePose(None, Rot, transl)

                    return True

            else:
                print ("\tRetrieve 3D_points from all pair matches to pnp")
                [matching_AB.remove_outliers(self.Fundamentalmat, self.Essentialmat)  for matching_AB in matching_vector]
                [matching_AB.update_inliers_mask()  for matching_AB in matching_vector]

                inter_3d_pts, inter_2d_pts, inter_2d_pts_norm, mask = self.__points_to_pnp_multiple(matching_vector)

                print ("\tresection using pnp")
                Result = self.__estimateCameraPose_pnp(inter_3d_pts, inter_2d_pts, inter_2d_pts_norm, matching_vector[0].image_B.cameraMatrix)
                if Result is None: 
                    # pas interesant la nouvelle image
                    return False
                inliers_Number, Rot, transl, mask_inliers = Result

                """Set camera pose of new image"""
                matching_vector[0].image_B.setAbsolutePose(None, Rot, transl)

                return True


    def __points_to_pnp_singal(self, matching_AB):
        ''' Intersection 3D 2D '''
        Result = matching_AB.retrieve_existing_points()
        """ Not enough points """
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
            """ not enough points """
            if Result is None: 
                """ remove it : do not take it into account for triangulation """
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
    def __best_match(self, matches_vector):
        if configuration["pnpsolver_method"] and configuration["use_allimages"]:
            return matches_vector
            
        if configuration["pnpsolver_method"] and not configuration["use_allimages"]:
            # /***************************************************************/
            # /** ''' Retrieve best candidate pair for incremetal phase ''' **/
            # /***************************************************************/
            """ Retrieve a pair of images which one of them is already used """
            print("\timage {} --> number total matches homography {}".format(new_image.id, nb_matches_homog))
           
            index_candidate_pair = np.argmax([x.homog_mask_len for x in matches_vector])
            matching_AB = matches_vector[index_candidate_pair]
            
            # Rectify precedent_image, current_image
            if(matching_AB.image_B in self.last_images and matching_AB.image_A not in self.last_images):
                matching_AB.permute_prec_curr()

            print ("\n-Retrieve best candidate pair for incremetal phase: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")  

            return [matching_AB]

            
        if not configuration["pnpsolver_method"]:
            print("Not implemented yet")
            return None


    # /****************************************************************/
    # /*****    Function of display Point-Cloud and Camera pos    *****/          
    # /****************************************************************/
    def display_pose(self):
        """ retrieve all 3D-points to draw """
        p_cloud_list = [img.points_3D_used for img in self.last_images]
        p_cloud = union(p_cloud_list)

        print ("\n- Number of 3D points of 3D-reconstruction generated with SfM is  ",len(p_cloud), "points")

        """  Save point-cloud in ply file  """
        pts2ply(p_cloud, filename = "point_cloud.ply")
        
        """ Config viewer """
        vis = o3d.visualization.Visualizer()
        vis.create_window("Structure_from_Motion", 1280, 720)

        """  Add Camera-pose of images to viewer """
        scale = 0.5
        for img in (self.remaining_images):
            if("transform" in img.absoluteTransformation.keys()):

                mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scale).transform(img.absoluteTransformation["transform"])
                id_ref_imd_id = text_3d(img.id, img.absoluteTransformation["transform"][0:3, 3], degree = 180.)
                
                vis.add_geometry(mesh_img_i)
                vis.add_geometry(id_ref_imd_id)

        """  Add point-cloud of 3D-reconstruction to viewer """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(p_cloud).reshape(-1, 3))

        vis.add_geometry(pcd)

        """  Launch Visualisation  """
        vis.run()

