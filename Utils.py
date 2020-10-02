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