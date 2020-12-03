**This project is a pipeline of Structure from Motion (SfM) which use real datasets.**

**Author**: Rachid LADJOUZI for Wemap SAS  
**Keywords**: Matching, Triangulation, Bundle Adjustment, Homography, Relocalisation


# Installation
mySfM requires python 3.6 or 3.7 

    pip3 install -r requirements.txt

or

    python3 Setup.py install

# Usage

    python3 src/myProgram.py

This command:
- starts the cloud point reconstruction of `datasets/Saint_Roch_new/data` dataset,
- find pose of new images from `datasets/Saint_Roch_new/reloc` uses relocalisation.

# Use your own dataset

## 3D point cloud construction

- Modify the path of `self.ImageFolder` in [src/SfM.py](src/SfM.py) with your own path.  
- Include a *.yaml file in your dataset folder which contains the camera intrinsic parameters. An example is given in [datasets/Saint_Roch_new/data/camera_Samsung_s7.yaml]

## Relocalisation

- Modify the path of `self.ImageFolder` in [src/Reloc.py](src/Reloc.py) with your own path.  
- Include a *.yaml file in your dataset folder which contains the camera intrinsic parameters. An example is given in [datasets/Saint_Roch_new/data/camera_Samsung_s7.yaml]

