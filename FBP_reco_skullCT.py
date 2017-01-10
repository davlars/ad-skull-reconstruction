"""
FBP reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
from adutils import *

# Discretization
discr_reco_space = get_discretization()

# Forward operator (in the form of a broadcast operator)
A = get_ray_trafo()

# Define fbp
fbp = get_fbp(A)

# Data
rhs = get_data(A)

# Reconstruct
x = fbp(rhs)

# Save
saveName = '/lcrnas/data/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_FBP.npy'
np.save(saveName,np.asarray(x))
