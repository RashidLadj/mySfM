import cv2 as cv
 
def is_CV2():
  return cv.__version__.startswith("2.")

def is_CV3():
  return cv.__version__.startswith("3.")

def is_CV4():
  return cv.__version__.startswith("4.")
