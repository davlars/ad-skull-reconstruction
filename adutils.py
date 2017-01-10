"""
Utility function to read simulated skull CT data.

Also fetches correct geomtry for reconstruction and provides a fbp defintion.
"""
import odl
import numpy as np
import os
import sys
import glob
import shutil
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data', 'Simulated', '120kV')
fileStart = 'HelicalSkullCT_70100644Phantom_no_bed_'
nTurns = 23
PY3 = (sys.version_info > (3, 0))


def load_data_from_nas(nas_path):
    """Load all the needed data from the nas onto your local machine

    This makes loading files much faster.

    Usage on windows where nas is bound to "Z:\"

    python -c "import adutils; adutils.load_data_from_nas('Z:\\')"
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    nas_data_path = os.path.join(nas_path, 'Simulated', '120kV')
    if not os.path.exists(nas_path):
        raise IOError('Cannot find NAS data at {}'.format(nas_data_path))

    for filename in glob.glob(os.path.join(nas_data_path, '*.*')):
        shutil.copy(filename, data_path)


def get_discretization():
    # Set geometry for discretization
    volumeSize = np.array([230.0, 230.0, 141.0546875])
    volumeOrigin = np.array([-115.0, -115.0, 0])

    # Discretization parameters
    nVoxels = np.array([512, 512, 314])

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

    if not os.path.exists(data_path):
        raise IOError('Could not find files at {}, have you run '
                      'adutils.load_data_from_nas()'.format(data_path))

    for turn in turns:
        print("Loading geometry for turn number {} out of {}".format(turn + 1, nTurns))
        geomFile = os.path.join(data_path, (fileStart + 'Turn_' + str(turn) + '.geometry.p'))

        # Load pickled geometry (small workaround of incompatibility between Python2/Python3 in pickle)
        with open(geomFile, 'rb') as f:
            if PY3:
                geom = pickle.load(f, encoding='latin1')
            else:
                geom = pickle.load(f)

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

    if not os.path.exists(data_path):
        raise IOError('Could not find files at {}, have you run '
                      'adutils.load_data_from_nas()'.format(data_path))

    imagesTurn = []
    for turn in turns:
        print("Loading data for turn number {} out of {}".format(turn + 1, nTurns))
        dataFile = os.path.join(data_path, (fileStart + 'Dose150mGy_Turn_' + str(turn) + '.data.npy'))
        projections = np.load(dataFile).astype('float32')
        imagesTurn.append(-np.log(projections / 7910))

    rhs = A.range.element(imagesTurn)

    return rhs
