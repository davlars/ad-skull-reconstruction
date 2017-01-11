"""
Landweber reconstruction example for simulated Skull CT data
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
x = reco_space.zero()

# Run such that recosntruction from last iterations is saved (saveReco =True), or not (saveReco = False)
saveReco = False

omega = 0.005
niter = 100
odl.solvers.landweber(A, x, rhs, niter=niter, omega=omega, callback = callbackPrintIter)

if saveReco:
    saveName = '/home/user/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Landweber_' + str(niter) + 'iterations.npy'
    np.save(saveName,np.asarray(x))
