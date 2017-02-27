"""
CGLS reconstruction example for simulated Skull CT data
"""
import odl
import numpy as np
import os
import adutils

rebin_factor = 10

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_rebin=True, rebin_factor=rebin_factor)

# Data
rhs = adutils.get_data(A, use_rebin=True, rebin_factor=rebin_factor)

# Reconstruct
callbackShowReco = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(coords=[None, 0, None]) &
                    odl.solvers.CallbackShow(coords=[0, None, None]) &
                    odl.solvers.CallbackShow(coords=[None, None, 60]))

callbackPrintIter = odl.solvers.CallbackPrintIteration()


title = 'my reco'
lamb = 0.01
pth = '/home/davlars/ad-skull-reconstruction/data/results/tv/lambda_{}'.format(float(lamb)) + '_iterate_{}.png'

# Reconstruct
callbackPrintIter = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(title, coords=[None, 0, None], clim=[0.018, 0.022]) &
            odl.solvers.CallbackShow(title, coords=[0, None, None], clim=[0.018, 0.022]) &
            odl.solvers.CallbackShow(title, coords=[None, None, 60], clim=[0.018, 0.022],
                                     saveto=pth))


callbackPrintIter = odl.solvers.CallbackPrintIteration()

# Start with initial guess
x = adutils.get_initial_guess(reco_space)

# Run such that the solution is saved to local repo (saveReco = True), or not (saveReco = False)
saveReco = True

niter = 2
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackPrintIter)

if saveReco:
    saveName = '/home/davlars/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_CGLS_' + str(niter) + 'iterations'
    adutils.save_data(x, saveName)
