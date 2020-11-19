###########################################################
##               Author : Rachid LADJOUZI                ##
###########################################################
from SfM import *


# /*******************************************************/
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

