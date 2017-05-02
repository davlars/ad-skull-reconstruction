"""
FBP reconstruction example for simulated Skull CT data
"""

import numpy as np
import adutils

# Discretization
reco_space = adutils.get_discretization(use_2D=True)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_2D=True)

# Define fbp
fbp = adutils.get_fbp(A, use_2D=True)

# Data
rhs = adutils.get_data(A, use_2D=True)

# Reconstruct
x = fbp(rhs)

# Show result
x.show()

# Compare to phantom
phantom = reco_space.element(adutils.get_phantom(use_2D=True))

phantom.show()

# Save
saveReco = True
if saveReco:
    saveName = '/home/davlars/reference_reconstruction_512_512.npy'
    np.save(saveName, np.asarray(x))
