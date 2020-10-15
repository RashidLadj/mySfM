import numpy as np
import yaml


""" TODO: read cameraparams from EXIF File """
""" Save Camera parameters in YAML file """
def save_cameraParams(pathFile):
    """ Save Camera parameters in YAML file"""
    extrinsic_params = dict(extrinsic_params = [[1.819682369480060515e+03,0.000000000000000000e+00,9.161596376751100479e+02],
                                                [0.000000000000000000e+00,1.812444689491887175e+03,5.307331980598187329e+02],
                                                [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
    distortion_coeff = dict(distortion_coeff = [None, None, None, None, None])
    resolution = dict(resolution = [1920, 1080])
    params = [extrinsic_params, distortion_coeff, resolution]

    camera_params = {}
    camera_params = dict(camera_params, **extrinsic_params)
    camera_params = dict(camera_params, **distortion_coeff)
    camera_params = dict(camera_params, **resolution)

    ## create YAML File ##
    with open(pathFile, 'w') as f:
        yaml.dump(camera_params, f)


""" Load Camera parameters from YAML file """
def load_cameraParams(pathFile):
   
    with open(pathFile) as f:
        camera_params = yaml.safe_load(f)

    extrinsic_params = np.array(camera_params["extrinsic_params"])
    distortion_coeff = np.array(camera_params["distortion_coeff"])
    resolution = np.array(camera_params["resolution"])

    return camera_params