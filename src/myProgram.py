###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################
from SfM import *
from Reloc import *

# /*****************  3D-Reconstruction  *****************/
# /*********       PHASE One: prepare Data       *********/   
# /*******        PHASE Two: Initialize SfM        *******/   
# /******        PHASE Three: Incremental SfM       ******/  
# /*****       PHASE Four: Display Point-Cloud       *****/          
# /*******************************************************/
structure_from_motion = SfM()
structure_from_motion.loadData()
structure_from_motion.init()
structure_from_motion.incremental()
structure_from_motion.save_data_for_BA()## just write data from file, and excute BA manualy
structure_from_motion.display_reconstruction()


# /******************  Relocalization  *******************/
# /*********       PHASE One: prepare Data       *********/   
# /*******        PHASE Two: Compute poses         *******/         
# /*******************************************************/
reloc = Reloc(structure_from_motion.last_images)
reloc.loadData()
reloc.display_pose()


