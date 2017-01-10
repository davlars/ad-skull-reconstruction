"""
Utility function to read simulated skull CT data and get correct geomtry for reconstruction, as well as an fbp defintion (from odl)
"""
import odl
import numpy as np
import os
import nibabel as nib
import pickle


filePath = '/lcrnas/data/Simulated/120kV/'
fileStart = 'HelicalSkullCT_70100644Phantom_no_bed_'
nTurns = 23


def getPhantomSize(phantomName):
    # nifit data
    path = '/lcrnas/data/Simulated/code/AD_GPUMCI/phantoms/'
    nii = nib.load(path+phantomName)
    phantomdata = nii.get_data()
    phantomSize = np.shape(phantomdata)
    return phantomSize


def get_discretization():
    phantomName = '70100644Phantom_labelled_no_bed.nii'

    # Set geometry for discretization
    volumeSize = np.array([230.0, 230.0, 141.0546875])
    volumeOrigin = np.array([-115.0, -115.0, 0])

    # Discretization parameters
    phantomSize = getPhantomSize(phantomName)
    nVoxels = np.array(phantomSize)

    # Discrete reconstruction space
    reco_space = odl.uniform_discr(volumeOrigin,
                                   volumeOrigin + volumeSize,
                                   nVoxels, dtype='float32')
    return reco_space


def get_ray_trafo(reco_space, use_subset=False):
    # Geometry and forward projector
    if use_subset:
        turns = range(13, 16)
    else:
        turns = range(nTurns)

    Aops = []

    for turn in turns:
        print("Loading geometry for turn number {} out of {}".format(turn + 1, nTurns))
        geomFile = os.path.join(filePath, (fileStart + 'Turn_' + str(turn) + '.geometry.p'))
        # Load pickled geometry (small workaround of incompatibility between Python2/Python3 in pickle)
        with open(geomFile, 'rb') as f:
            geom = pickle.load(f, encoding='latin1')
        # X-ray transform
        Aops.append(odl.tomo.RayTransform(reco_space, geom, impl='astra_cuda'))

    A = odl.BroadcastOperator(*Aops)

    return A


def get_fbp(A):
    fbp = odl.ReductionOperator(*[(odl.tomo.fbp_op(Ai,
                                              padding=True,
                                              filter_type='Hamming', #Hann
                                              frequency_scaling=0.8) *
                               odl.tomo.tam_danielson_window(Ai))
                              for Ai in A])
    return fbp


def get_data(A, use_subset=False):
    # Data
    if use_subset:
        turns = range(13, 16)
    else:
        turns = range(nTurns)

    imagesTurn = []
    for turn in turns:
        print("Loading data for turn number {} out of {}".format(turn + 1, nTurns))
        dataFile = os.path.join(filePath, (fileStart + 'Dose150mGy_Turn_' + str(turn) + '.data.npy'))
        projections = np.load(dataFile).astype('float32')
        imagesTurn.append(-np.log(projections / 7910))

    rhs = A.range.element(imagesTurn)

    return rhs






