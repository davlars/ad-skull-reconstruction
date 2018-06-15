# -*- coding: utf-8 -*-
"""
Example of loading data from imagingnas. 
"""

import numpy as np
import adutils 

#
#Define which dataset to load
#
#Note: '70100644 is the default one, and the only when where there is a 2D example
#
phantom_number = '70100644'
#phantom_number = '70114044'
#phantom_number = '70122044'
#phantom_number = '70135144'
#phantom_number = '70141544'
#phantom_number = '70153744'
#phantom_number = '70162244'

# Set nas path to Reference folder
nas_path = '/mnt/imagingnas/Reference'

# Set boolean to load data
load_data = True

# Set boolean to load phantoms
load_phantom = True

adutils.load_data_from_nas(nas_path, 
                           load_data=load_data, 
                           load_phantom=load_phantom, 
                           phantom_number=phantom_number)    


