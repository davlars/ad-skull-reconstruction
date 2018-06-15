"""
FBP reconstruction example for simulated Skull CT data
"""

import numpy as np
import adutils

# Define phantom name (or use default '70100644')
phantom_number = '70100644'

# Discretization
reco_space = adutils.get_discretization(phantom_number=phantom_number)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, 
                          use_subset=True, 
                          phantom_number=phantom_number)

# Define fbp
fbp = adutils.get_fbp(A)

# Data
rhs = adutils.get_data(A, 
                       use_subset=True, 
                       phantom_number=phantom_number)

# Reconstruct
x = fbp(rhs)

# Show result
x.show(coords=[None, 0, None])
x.show(coords=[0, None, None])
x.show(coords=[None, None, 90])

# Save
saveReco = False
if saveReco:
    saveName = '/home/user/FBP_reco.npy'
    adutils.save_image(x, 
                       saveName)
