"""
CGLS reconstruction example for simulated Skull CT data
"""
import odl
import numpy as np
import os
import adutils

# Define phantom name (or use default '70100644')
phantom_number = '70100644'

# Rebin data
rebin_factor = 10
adutils.rebin_data(rebin_factor, 
                   phantom_number=phantom_number)

# Discretization
reco_space = adutils.get_discretization(phantom_number=phantom_number)

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, 
                          phantom_number=phantom_number,
                          use_rebin=True, 
                          rebin_factor=rebin_factor)

# Data
rhs = adutils.get_data(A, 
                       phantom_number=phantom_number,
                       use_rebin=True, 
                       rebin_factor=rebin_factor)

# Reconstruct
title = 'my reco'
lamb = 0.01
pth = '/home/user/ad-skull-reconstruction/data/results/tv/lambda_{}'.format(float(lamb)) + '_iterate_{}.png'

callbackPrintIter = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(title, coords=[None, 0, None], clim=[0.018, 0.022]) &
            odl.solvers.CallbackShow(title, coords=[0, None, None], clim=[0.018, 0.022]) &
            odl.solvers.CallbackShow(title, coords=[None, None, 60], clim=[0.018, 0.022],
                                     saveto=pth))


callbackPrintIter = odl.solvers.CallbackPrintIteration()

# Start with initial guess, so far only for '70100644'
if phantom_number == '70100644':
    x = adutils.get_initial_guess(reco_space)
else:
    x = x = reco_space.zero()

# Run such that the solution is saved to local repo (saveReco = True), or not (saveReco = False)
saveReco = False

niter = 2
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackPrintIter)

if saveReco:
    saveName = '/home/user/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_CGLS_' + str(niter) + 'iterations'
    adutils.save_image(x, saveName)
