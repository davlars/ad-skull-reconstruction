"""
CGLS reconstruction example for simulated Skull CT data
"""
import odl
import numpy as np
import os
import adutils

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space)

# Data
rhs = adutils.get_data(A)

# Reconstruct
callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(coords=[None, 0, None]) &
                    odl.solvers.CallbackShow(coords=[0, None, None]) &
                    odl.solvers.CallbackShow(coords=[None, None, 60]))

callbackPrintIter = odl.solvers.CallbackPrintIteration()

# Start with empty x
x = adutils.get_initial_guess(reco_space)

# Run such that the solution is saved to local repo (saveReco = True), or not (saveReco = False)
saveReco = False

niter = 100
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackPrintIter)

if saveReco:
    saveName = '/home/user/data/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_CGLS_' + str(niter) + 'iterations.npy'
    np.save(saveName,np.asarray(x))
