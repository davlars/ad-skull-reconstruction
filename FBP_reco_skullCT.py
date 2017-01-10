"""
FBP reconstruction example for simulated Skull CT data
"""

import numpy as np
import adutils

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space)

# Define fbp
fbp = adutils.get_fbp(A)

# Data
rhs = adutils.get_data(A)

# Reconstruct
x = fbp(rhs)

# Save
if False:
    saveName = '/lcrnas/data/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_FBP.npy'
    np.save(saveName, np.asarray(x))
