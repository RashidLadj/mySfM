import numpy as np
import cv2 as cv
import open3d as o3d

from pyquaternion import Quaternion
from PIL import Image as Im
from PIL import ImageFont, ImageDraw

import open3d as o3d

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#########################################################################
## https://gist.github.com/RashidLadj/bac71f3d3380064de2f9abe0ae43c19e ##
#########################################################################
def intersect2D(Array_A, Array_B):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows and index of items in A & B arrays
  """
  # [IDX, IDY] where Equal
  IndexEqual = np.asarray([(i, j, tuple(x)) for i,x in enumerate(Array_A) for j, y in enumerate (Array_B) if( np.array_equal(x, y) )])

  idx = IndexEqual[:, 0].astype(int) if len(IndexEqual) != 0 else []
  idy = IndexEqual[:, 1].astype(int) if len(IndexEqual) != 0 else []

  intersectionList = IndexEqual[:, 2] if len(IndexEqual) != 0 else []

  return intersectionList, idx, idy


def union (myList, dimm = 3):
    union = set([])
    for Element in myList:
        Element  = set([tuple(x) for x in list(Element)])
        union = (union | Element)
    return np.array([list(x) for x in union])


#####################################################################################################################################
##                                               reprjection Error Formula                                                         ##
## https://stackoverflow.com/questions/23781089/opencv-calibratecamera-2-reprojection-error-and-custom-computed-one-not-agree?rq=1 ##
#####################################################################################################################################
def compute_reprojection_error_2(transfom_matrix, pts_3d, pts_2d, camera_matrix = np.eye(3)):
    rVec, tVec = r_and_t_Vec_from_transformation(transfom_matrix)
    return compute_reprojection_error_1(rVec, tVec,pts_3d, pts_2d, camera_matrix)
    

def compute_reprojection_error_1(rVec, tVec, pts_3d, pts_2d, camera_matrix):
    project_points, _ = cv.projectPoints(pts_3d, rVec, tVec, camera_matrix, distCoeffs=None)
    project_points = project_points.reshape(-1, 2)

    reprojection_error_avg = sum([np.linalg.norm(pts_2d[i] - project_points[i])  for i in range (len(project_points))]) / len(project_points)
    
    return reprojection_error_avg
    


# # # # # # # # # # # # # # # # # #
# #     R | s*t                 # # 
# # T = -------                 # # 
# #     0 0 0 1                 # #     
# #                             # #
# #           R.T | -R.T*s*t    # # 
# # inv(T) =  --------------    # # 
# #            0  0  0  1       # # 
# #                             # #
# # P = R.T| -R.T*s*t           # # 
# # P = inv(T)[:3, :4]          # # 
# # # # # # # # # # # # # # # # # #

####################################################################
##      [4x4] homogeneous Transform from [3x3] R and [3x1] t      ##
####################################################################  
def poseRt(Rot, transl):
    """ Transformation matrix (Homogenous)  """
    trans = np.eye(4)
    trans[:3, :3] = Rot[:3, :3]
    trans[:3, 3]  = transl.reshape(-1)

    return trans


# [4x4] homogeneous inverse T^-1 from [4x4] T     
def inv_poseRt(Trans):
    """ ret == np.linalg.inv(Trans) """
    ret = np.eye(4)
    R_T = Trans[:3,:3].T
    t   = Trans[:3,3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ t

    return ret 

# [4x3] Matrix inverse T^-1[3, 4] from [4x4] T 
def projection_from_transformation(Trans):
    return inv_poseRt(Trans)[:3, :4]


def current_relative_transform(current_absolute_transform, precedent_absolute_transform):    
    # transform_AC = transform_BC @ transform_AB
    # ==> transform_BC = transform_AC * inv(transform_AB)
    return current_absolute_transform @ inv_poseRt(precedent_absolute_transform)


def r_and_t_Vec_from_transformation(transform):   
    proj = projection_from_transformation(transform)
    rVec = cv.Rodrigues(transform[:3, :3])[0].reshape(-1)
    tVec = proj[:3, 2]
    return rVec, tVec


#/***********************************************/#
# Function: Check Rotation Matrix (Must be det=1) #
#/***********************************************/#
def CheckCoherentRotation(Rotation_mat):
    if np.abs( np.linalg.det( Rotation_mat ) - 1 > 1e-07):
        print ("det(R) != +-1.0, this is not a rotation matrix")
        return False
    return True


################## From solvePNP ###########################
# Projection = np.concatenate(cv.rodriges(rvec)[0], tvec)  #
# Rot.T = cv.rodriges(rvec)[0]                             # 
# - Rot.T * transl = tvec                                  #
#############################################################



#/**************************************************************/#
#  """Saves an ndarray of 3D coordinates (in meshlab format)"""  #
#/**************************************************************/#
def pts2ply(pts, colors = None, filename = 'point_cloud.ply'): 
    with open(filename,'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')

        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')

        if colors == None:
            colors = np.zeros((len(pts), 3), dtype=int)
            colors[:, 0] = 255
        colors.astype(int)
        
        for pt, color in zip(pts, colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0],   pt[1],   pt[2],
                                                 color[0],color[1],color[2]))


def text_3d(text, pos, direction=(1., 0., 0.), degree=0., font='/Library/Fonts/Arial.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Im.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 200.0)

    raxis = np.cross([0.0, 0.0, 0.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

