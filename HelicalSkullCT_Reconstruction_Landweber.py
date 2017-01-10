"""
Landweber reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
import os
import nibabel as nib
import pickle 

def getPhantomSize(phantomName):
    #nifit data 
    path = '/lcrnas/data/Simulated/code/AD_GPUMCI/phantoms/'
    nii = nib.load(path+phantomName)
    phantomdata = nii.get_data()
    phantomSize = np.shape(phantomdata)
    return phantomSize

# Input parameters    
filePath = '/lcrnas/data/Simulated/120kV/'
fileStart = 'HelicalSkullCT_70100644Phantom_no_bed_'
phantomName = '70100644Phantom_labelled_no_bed.nii'

# Set geometry for discretization
pitch_mm = 6.6
nTurns = 23
volumeSize = np.array([230.0, 230.0, 141.0546875])
volumeOrigin = np.array([-115.0, -115.0, 0]) 

# Discretization parameters
phantomSize = getPhantomSize(phantomName)
nVoxels, nPixels = np.array(phantomSize), [500, 20]

# Discrete reconstruction space
discr_reco_space = odl.uniform_discr(volumeOrigin,volumeOrigin + volumeSize,
                                     nVoxels, dtype='float32')

# Geometry and forward projector
turns = range(nTurns)
Aops = []
for turn in turns:
    print("Loading geometry for turn number %i out of %i" % (turn+1,nTurns))
    geomFile = os.path.join(filePath,(fileStart + 'Turn_' + str(turn) + '.geometry.p'))
    # Load pickled geometry (small workaround of incompatibility between Python2/Python3 in pickle)
    with open(geomFile, 'rb') as f:
        geom = pickle.load(f, encoding='latin1')
    # X-ray transform
    Aops += [odl.tomo.RayTransform(discr_reco_space, geom, impl='astra_cuda')]

A = odl.BroadcastOperator(*Aops)

# Data
imagesTurn = []
for turn in turns:   
    print("Loading data for turn number %i out of %i" % (turn+1,nTurns))
    dataFile = os.path.join(filePath,(fileStart + 'Dose150mGy_Turn_' + str(turn) + '.data.npy'))
    projections = np.load(dataFile).astype('float32')
    imagesTurn += [-np.log(projections / 7910)] 

rhs = A.range.element(imagesTurn)

# Reconstruct
callbackShowReco = (odl.solvers.CallbackPrintIteration() & 
            odl.solvers.CallbackShow(coords=[None, 0, None]) & 
            odl.solvers.CallbackShow(coords=[0, None, None]) & 
            odl.solvers.CallbackShow(coords=[None, None, 60]))

callbackPrintIter = odl.solvers.CallbackPrintIteration()

# Start with empty x
x = discr_reco_space.zero()

# Run such that every 5th iteration is saved (saveCont == 1) or only the last one (saveCont == 0)
saveCont = 0

omega = 0.01

if saveCont == 0:
    niter = 5
    odl.solvers.landweber(A, x, rhs, niter=niter, omega=omega, callback = callbackPrintIter)
    saveName = os.path.join(filePath,'reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Landweber_' + 
                                      str(niter) + 'iterations.npy')
    np.save(saveName,np.asarray(x))

else:    
    startiter = 5
    enditer = 101
    stepiter = 5
    niter = [int(i) for i in np.arange(startiter,enditer,stepiter)]
    saveNameStart = 'Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_Landweber_'
    savePath = os.path.join(filePath,'reco',saveNameStart)
    for iterations in niter:
        odl.solvers.landweber(A, x, rhs, niter=stepiter, omega=omega, callback = callbackPrintIter)
        saveName = (savePath + '{}iterations'.format(iterations) + '.npy')
        np.save(saveName,np.asarray(x))
