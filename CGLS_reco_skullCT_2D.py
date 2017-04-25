"""
CGLS reconstruction example for simulated Skull CT data
"""
import odl
import numpy as np
import adutils

# Discretization
reco_space = adutils.get_discretization(use_2D=True)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_2D=True)

# Data
rhs = adutils.get_data(A, use_2D=True)

# Reconstruct
title = 'my reco'

callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(clim=[0.018, 0.022]))

callbackPrintIter = (odl.solvers.CallbackPrintIteration())

# Start with initial guess
x = adutils.get_initial_guess(reco_space)
#x = A.domain.zero()

# Run such that the solution is saved to local repo (saveReco = True), or not (saveReco = False)
saveReco = False

niter = 30
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback=callbackShowReco)

if saveReco:
    saveName = '/home/davlars/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_CGLS_' + str(niter) + 'iterations'
    adutils.save_data(x, saveName)
