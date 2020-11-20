import numpy as np
import cv2 as cv 

class Landmark:
    def __init__(self, id, pos_3D):
        self.id = id
        self.pos_3D = list(pos_3D)
        self.images = []
        self.pos_2D = []

    def add_image(self, image, pos_2D):
        self.image = image
        self.images.append(image)
        self.pos_2D.append(pos_2D)

    def set_pos_3D(self, pos_3D):
        self.pos_3D = list(pos_3D)

        for i, image in enumerate(self.images):
            idx = (image.points_2D_used.tolist()).index(self.pos_2D[i].tolist())   
            image.points_3D_used[idx] = list(pos_3D)       


class P_cloud:
    def __init__(self, images_list):
        self.landmarks = np.empty((0,), dtype=Landmark)
        self.id_pt_3D = 0
        self.images_list = images_list
        self.__config()
        # self.generate_file()

    
    def __config(self):
        for image in self.images_list:
            for pt_2D, pt_3D in zip (image.points_2D_used, image.points_3D_used):
                self.__add_pt(pt_3D, image, pt_2D)


    def __add_pt(self, pt_3D, img_ID, pt_2D):
        pt_3D_list = [ landmark.pos_3D for landmark in self.landmarks ]
        
        if list(pt_3D) in pt_3D_list:
            # print(existe)
            idx = pt_3D_list.index(list(pt_3D))
            self.landmarks[idx].add_image(img_ID, pt_2D)

        else:
            landmark = Landmark(self.id_pt_3D, pt_3D)
            landmark.add_image(img_ID, pt_2D)
            self.landmarks = np.append(self.landmarks, landmark)
            self.id_pt_3D += 1
        
        
    def generate_file(self):
        # create file
        file = open("Tests//Bundle_Adjustment/BA_Data.txt","w") 

        # First line --> { NB_Image, NB_3D_Points, NB_2D_Points }
        nb_observation = sum([len(landmark.pos_2D) for landmark in self.landmarks ])
        first_line = "{} {} {}\n".format(len(self.images_list), len(self.landmarks), nb_observation)
        file.write(first_line)
        
        for i, image in enumerate(self.images_list):
            image.BA_id = i

        # second step
        for landmark in self.landmarks:
            for i, _ in enumerate(landmark.images):
                data = "{} {} {} {}\n".format(landmark.images[i].BA_id, landmark.id,  landmark.pos_2D[i][0], landmark.pos_2D[i][1])
                file.write(data)

        # Third Step : Camera parameters
        for image in self.images_list:
            rVec = cv.Rodrigues( image.absoluteTransformation["rotation"].T )[0].ravel()
            rot_data = "{} {} {} ".format(rVec[0], rVec[1], rVec[2])
            file.write(rot_data)

            tVec = image.absoluteTransformation["translation"].ravel()
            tvec = (- np.linalg.inv(image.absoluteTransformation["rotation"]) @ image.absoluteTransformation["translation"]).ravel()
            transl_data = "{} {} {} ".format(tvec[0], tvec[1], tvec[2])
            file.write(transl_data)

            # intrinsic_data = "{} \n0.0 \n0.0\n".format(image.cameraMatrix[0][0])
            intrinsic_data = "{} {} {} {}\n".format(image.cameraMatrix[0][0], image.cameraMatrix[1][1], image.cameraMatrix[0][2], image.cameraMatrix[1][2])
            file.write(intrinsic_data)

        
        for landmark in self.landmarks:
            coor_3D = "{} {} {}\n".format(landmark.pos_3D[0], landmark.pos_3D[1], landmark.pos_3D[2])
            file.write(coor_3D)
            
        # Close File
        file.close()


