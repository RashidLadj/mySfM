import numpy as np
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
  # IndexEqual = np.asarray([(i, j, tuple(x)) for i,x in enumerate(Array_A) for j, y in enumerate (Array_B) if( (x[0] - y[0] >= -1) &  (x[0] - y[0] <= 1) &  (x[1] - y[1] >= -1) &  (x[1] - y[1] <= 1) )])

  IndexEqual = np.asarray([(i, j, tuple(x)) for i,x in enumerate(Array_A) for j, y in enumerate (Array_B) if( tuple(x) == tuple(y) )])


  idx = IndexEqual[:, 0].astype(int) if len(IndexEqual) != 0 else []
  idy = IndexEqual[:, 1].astype(int) if len(IndexEqual) != 0 else []

  intersectionList = IndexEqual[:, 2] if len(IndexEqual) != 0 else []

  return intersectionList, idx, idy


#####################################################################################################################################
##                                               reprjection Error Formula                                                         ##
## https://stackoverflow.com/questions/23781089/opencv-calibratecamera-2-reprojection-error-and-custom-computed-one-not-agree?rq=1 ##
#####################################################################################################################################


#####################################################################################################################################
##                                               Filter negative depth source                                                      ##
##                    https://github.com/xdspacelab/openvslam/blob/master/src/openvslam/camera/perspective.cc#L155                 ##
#####################################################################################################################################

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


################## From solvePNP ###########################
# Projection = np.concatenate(cv.rodriges(rvec)[0], tvec)  #
# Rot.T = cv.rodriges(rvec)[0]                             # 
# - Rot.T * transl = tvec                                  #
#############################################################
