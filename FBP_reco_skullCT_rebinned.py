"""
FBP reconstruction example for simulated Skull CT data using rebinned data
"""

import numpy as np
import adutils

# Define phantom name (or use default '70100644')
phantom_number = '70100644'

rebin_factor = 10
adutils.rebin_data(rebin_factor, 
                   phantom_number=phantom_number)

# Discretization
reco_space = adutils.get_discretization(phantom_number=phantom_number)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, 
                          phantom_number=phantom_number, 
                          use_rebin=True, 
                          rebin_factor=rebin_factor)

# Define fbp
fbp = adutils.get_fbp(A)

# Data
rhs = adutils.get_data(A, 
                       phantom_number=phantom_number,
                       use_rebin=True, 
                       rebin_factor=rebin_factor)

# Reconstruct
x = fbp(rhs)

# Show result
x.show(coords=[None, 0, None])
x.show(coords=[0, None, None])
x.show(coords=[None, None, 90])

# Save
saveReco = False
if saveReco:
    saveName = '/home/user/FBP_reco_rebinned.npy'
    adutils.save_image(x, 
                       saveName)