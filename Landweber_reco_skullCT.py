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

# Run such that every 5th iteration is saved (saveCont == True)
# or only the last one (saveCont == False)
saveCont = 0

omega = 0.005

if not saveCont:
    niter = 5
    odl.solvers.landweber(A, x, rhs, niter=niter, omega=omega, callback = callbackPrintIter)
    if False:
        saveName = '/lcrnas/data/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Landweber_' + str(niter) + 'iterations.npy'
        np.save(saveName,np.asarray(x))
else:
    startiter = 5
    enditer = 101
    stepiter = 5
    niter = [int(i) for i in np.arange(startiter,enditer,stepiter)]
    saveNameStart = 'Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Landweber_'
    savePath = os.path.join('/lcrnas/data/Simulated/120kV/','reco',saveNameStart)
    for iterations in niter:
        odl.solvers.landweber(A, x, rhs, niter=stepiter, omega=omega,
                              callback=callbackPrintIter)
        if False:
            saveName = (savePath + '{}iterations'.format(iterations) + '.npy')
            np.save(saveName,np.asarray(x))
