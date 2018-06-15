# -*- coding: utf-8 -*-
"""
Example of call to plot_data. 
"""

import numpy as np
import adutils 

data = adutils.get_phantom() #Load phantom data

adutils.plot_data(data,plot_separately=False) #Plot phantom in relevant cuts
    



