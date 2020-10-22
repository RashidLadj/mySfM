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
        self.cameraParams = load_cameraParams(self.ImageFolder+"camera_Samsung_s7.yaml")

    
    def loadData(self):
        assert path.exists(self.ImageFolder), 'veuillez verifier le chemin du Folder'
        images_name = sorted([file for file in listdir(self.ImageFolder) if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".PNG") or file.endswith(".png")])


        ###########################################################
        ##                  ''' Create Images '''                ##
        ###########################################################
        print ("\nRead images")
        self.images = []
        for image in images_name:
            self.images.append(Image_(self.ImageFolder, image, configuration["feature_process_size"], copy.deepcopy(self.cameraParams)))
        print ("    Images = ", images_name)


        ###########################################################
        ##       ''' Calculate KeyPoint and Descriptor '''       ##
        ###########################################################
        print ("\nDetect and compute descriptors")
        for image in self.images:
            image.keyPointDetector(self.detector)
            print ('\tsize of descriptors of', image.id, image.des.shape)
            

        ###########################################################
        ##  ''' Compute Matching beetwin each pair of image '''  ##
        ###########################################################
        print ("\nMatching points beetwin each pair of image")

        image_Pair = list(itertools.combinations(self.images, 2))

        self.matches_vector = []
        for image_A, image_B in image_Pair:
            matching_AB = Matching(self.matchingConfig, image_A, image_B)
            Homog_mask = self.Homographymat.compute_HomographyMatrix(matching_AB)
            matching_AB.setHomog_mask(np.sum(Homog_mask))
            self.matches_vector.append(matching_AB)
        

    def init(self):

        ###############################################################
        ##  ''' Retrieve best candidate pair for initialization '''  ##
        ###############################################################

        # print  ("Phase Init :")
        # print  ("\tImages disponibles :")
        # [print ("\t\t",x.id) for x in self.images]
        # print  ("\tImages utilisées :")
        # [print ("\t\t",x.id) for x in self.last_images]
        # print  ("\tmatches disponibles :")
        # [print ("\t\t",x.image_A.id,"  ", x.image_B.id, "   ", x.homog_mask_len ) for x in self.matches_vector]

        index_candidate_pair = np.argmax([x.homog_mask_len for x in self.matches_vector])
        matching_AB = self.matches_vector[index_candidate_pair]
        self.matches_vector.remove(matching_AB)
       
        print ("\n-Retrieve best candidate pair for initialization: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")

        ################################################################
        ##           ''' Compute Transformation Matrix '''            ##
        ##      ''' Triangulation and Initialize point-Cloud '''      ##
        ################################################################
        print ("\tCompute transformation and initialize point-Cloud")
        matching_AB.computePose_2D2D(self.Fundamentalmat, self.Essentialmat, initialize_step = True)
       
        self.last_images.append(matching_AB.image_A)
        self.last_images.append(matching_AB.image_B)

        self.images.remove(matching_AB.image_A)
        self.images.remove(matching_AB.image_B)


    def incremental(self):
        while(len(self.matches_vector) > 0):
            matching_AB = self.best_new_image()
            
            ################################################################
            ##       ''' Compute Transformation Matrix using PNP '''      ##
            ##       ''' Triangulation and increment point-Cloud '''      ##
            ################################################################

            if not(matching_AB == None):
                if configuration["incremental_method_is_resection"] == False:
                    matching_AB.computePose_2D2D_Scale(self.Essentialmat)
                else:
                    done = matching_AB.computePose_3D2D(self.Fundamentalmat, self.Essentialmat)
                    
                    if done:
                        self.last_images.append(matching_AB.image_B)
                        self.images.remove(matching_AB.image_B)
                    else:
                        print ("Resection --> not possible")

    
    def best_new_image(self):
        # /***************************************************************/
        # /** ''' Retrieve best candidate pair for incremetal phase ''' **/
        # /***************************************************************/
        
        
        """ recuprer les pair d'image ou une des deux images à déja été utilisée """
        [ self.matches_vector.remove(x) for x in (self.matches_vector) if ((x.image_A in self.last_images) and (x.image_B in self.last_images)) ]

        if len(self.matches_vector) == 0:
            #Normalement on y entre pas
            print(" ------- Bug a corriger !!! ------- ")
            return None
            
        # /** ^ == XOR **/
        """ Recuperer la meilleures images qui partagent le plus de matches ( avec le mask d'homography) avec l'une des images qui a été deja été utilisée """
        matches_vector = [x for x in self.matches_vector if ((x.image_A in self.last_images) ^ (x.image_B in self.last_images))]

        # print ("Phase Incremental :")
        # print ("\tImages disponibles :")
        # [print("\t\t",x.id) for x in self.images]
        # print ("\tImages utilisées :")
        # [print("\t\t",x.id) for x in self.last_images]
        # print ("\tmatches disponibles tout:")
        # [print("\t\t",x.image_A.id,"  ", x.image_B.id, "   ", x.homog_mask_len ) for x in self.matches_vector]
        # print ("\tmatches disponibles selectionné:")
        # [print("\t\t",x.image_A.id,"  ", x.image_B.id, "   ", x.homog_mask_len ) for x in matches_vector]

        if len(matches_vector) == 0:
            print("\nles deux images on déjà été traité !!!")
            return None

        """ je check le mask de l'homographie """
        if configuration["pnpsolver_method"] and configuration["use_allimages"]:
            print("je dois implementer")
            return None

        if configuration["pnpsolver_method"] and not configuration["use_allimages"]:
           
            index_candidate_pair = np.argmax([x.homog_mask_len for x in matches_vector])
            matching_AB = matches_vector[index_candidate_pair]
            self.matches_vector.remove(matching_AB)
            
            # Rectify precedent_image, current_image
            if(matching_AB.image_B in self.last_images and matching_AB.image_A not in self.last_images):
                matching_AB.permute_prec_curr()

            print ("\n-Retrieve best candidate pair for incremetal phase: (",matching_AB.image_A.id, ",", matching_AB.image_B.id, ") with ", len(matching_AB.matches), "matches")  

            return matching_AB

            
        if not configuration["pnpsolver_method"]:
            print("je dois implementer")
            return None


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
            mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scale).transform(img.absoluteTransformation["transform"])
            id_ref_imd_id = text_3d(img.id, img.absoluteTransformation["transform"][0:3, 3])
            
            vis.add_geometry(mesh_img_i)
            vis.add_geometry(id_ref_imd_id)

        """  Add point-Cloud of 3D-reconstruction to viewer """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(p_cloud).reshape(-1, 3))

        vis.add_geometry(pcd)

        """  Launch Visualisation  """
        vis.run()            