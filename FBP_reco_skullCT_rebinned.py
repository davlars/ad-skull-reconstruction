"""
FBP reconstruction example for simulated Skull CT data using rebinned data
"""

import numpy as np
import adutils

rebin_factor = 10
adutils.rebin_data(rebin_factor)

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_rebin=True, rebin_factor=rebin_factor)

# Define fbp
fbp = adutils.get_fbp(A)

# Data
rhs = adutils.get_data(A, use_rebin=True, rebin_factor=rebin_factor)

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
    adutils.save_image(x, saveName)