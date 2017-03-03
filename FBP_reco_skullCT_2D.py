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
x.show(coords=[None, 0, None], clim=[0.010, 0.020])
x.show(coords=[0, None, None], clim=[0.010, 0.020])
x.show(coords=[None, None, 75], clim=[0.010, 0.020])

# Save
saveReco = False
if saveReco:
    saveName = '/home/user/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_FBP_2D.npy'
    np.save(saveName, np.asarray(x))
