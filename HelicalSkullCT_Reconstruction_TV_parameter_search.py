"""
TV reconstruction example for simulated Skull CT data - lambda parameter search
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

# Define fbp in order to use it as initial guess for TV reco
fbp = odl.ReductionOperator(*[(odl.tomo.fbp_op(Ai, 
                                              padding=True,
                                              filter_type='Hamming', #Hann 
                                              frequency_scaling=0.8) *
                               odl.tomo.tam_danielson_window(Ai))
                              for Ai in A])

# Data
imagesTurn = []
for turn in turns:   
    print("Loading data for turn number %i out of %i" % (turn+1,nTurns))
    dataFile = os.path.join(filePath,(fileStart + 'Dose150mGy_Turn_' + str(turn) + '.data.npy'))
    projections = np.load(dataFile).astype('float32')
    imagesTurn += [-np.log(projections / 7910)] 

rhs = A.range.element(imagesTurn)

# Gradient operator
gradient = odl.Gradient(discr_reco_space, method='forward')

# Column vector of operators
op = odl.BroadcastOperator(A, gradient)

Anorm = odl.power_method_opnorm(A[1], maxiter=2)
Dnorm = odl.power_method_opnorm(gradient, 
                                xstart=odl.phantom.white_noise(gradient.domain), 
                                maxiter=10)

# Estimated operator norm, add 10 percent
op_norm = 1.1 * np.sqrt(len(A.operators)*(Anorm**2) + Dnorm**2)

print('Norm of the product space operator: {}'.format(op_norm))

lambs = (0.01, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001)

# Use FBP as initial guess
x_init = fbp(rhs)

for lamb in lambs:      
    # l2-squared data matching
    l2_norm = odl.solvers.L2NormSquared(A.range).translated(rhs)

    # Isotropic TV-regularization i.e. the l1-norm
    l1_norm = lamb * odl.solvers.L1Norm(gradient.range)

    # Combine functionals
    f = odl.solvers.SeparableSum(l2_norm, l1_norm)
    
    # Set g functional to zero
    g = odl.solvers.ZeroFunctional(op.domain)
    
    # Accelerataion parameter
    gamma=0.4
    
    # Step size for the proximal operator for the primal variable x
    tau = 1.0 / op_norm 
    	
    # Step size for the proximal operator for the dual variable y
    sigma = 1.0 / op_norm #1.0 / (op_norm ** 2 * tau)

    # Reconstruct
    callbackShowReco = (odl.solvers.CallbackPrintIteration() & #Print iterations
                odl.solvers.CallbackShow(coords=[None, 0, None]) & #Show parital reconstructions
                odl.solvers.CallbackShow(coords=[0, None, None]) & 
                odl.solvers.CallbackShow(coords=[None, None, 60]))
    	
    callbackPrintIter = odl.solvers.CallbackPrintIteration()
             
    # Use the FBP as initial guess
    x = x_init
    
    # Run such that every 5th iteration is saved (saveCont == 1) or only the last one (saveCont == 0)
    saveCont = 1
    
    if saveCont == 0:
        niter = 100
        odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma = sigma, 
                                          niter = niter, gamma=gamma, callback=callbackPrintIter)
        saveName = os.path.join(filePath,'reco/Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_TV_lambda{}_'.format(lamb) + 
                                          str(niter) + 'iterations.npy')
        np.save(saveName,np.asarray(x))
        
    else:    
        startiter = 5
        enditer = 101
        stepiter = 5
        niter = [int(i) for i in np.arange(startiter,enditer,stepiter)]
        saveNameStart = 'Reco_HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_TV_lambda{}_'.format(lamb)
        savePath = os.path.join(filePath,'reco',saveNameStart)
        for iterations in niter:
            odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma = sigma, 
                                              niter = stepiter, gamma=gamma, callback=callbackPrintIter)
            saveName = (savePath + '{}iterations'.format(iterations) + '.npy')
            np.save(saveName,np.asarray(x))
