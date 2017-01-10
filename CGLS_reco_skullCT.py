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
x = reco_space.zero()

# Run such that every 5th iteration is saved (saveCont == True)
# or only the last one (saveCont == False)
saveCont = False

if not saveCont:
    niter = 5
    odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=niter, callback = callbackPrintIter)
    if False:
        saveName = '/home/user/data/Simulated/120kV/reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_CGLS_' + str(niter) + 'iterations.npy'
        np.save(saveName,np.asarray(x))

else:
    startiter = 5
    enditer = 101
    stepiter = 5
    niter = [int(i) for i in np.arange(startiter, enditer, stepiter)]
    for iterations in niter:
        odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=stepiter,
                                              callback=callbackPrintIter)
        if False:
            saveNameStart = 'Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_CGLS_'
            savePath = os.path.join('/home/user/Simulated/120kV/','reco',saveNameStart)
            saveName = (savePath + '{}iterations'.format(iterations) + '.npy')
            np.save(saveName, np.asarray(x))
