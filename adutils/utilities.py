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
import nibabel as nib
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data', 'Simulated', '120kV')

PHANTOM_NUMBERS = ['70100644', '70114044', '70122044', '70135144',
                   '70141544', '70153744', '70162244']
DEFAULT_PHANOM_NUMBER = PHANTOM_NUMBERS[0]
DEFAULT_REBIN_FACTOR = 10

nTurns = 23
PY3 = (sys.version_info > (3, 0))


__all__ = ('PHANTOM_NUMBERS',
           'DEFAULT_PHANOM_NUMBER',
           'DEFAULT_REBIN_FACTOR',
           'load_data_from_nas',
           'get_discretization',
           'get_ray_trafo',
           'get_fbp',
           'get_initial_guess',
           'get_data',
           'get_phantom',
           'plot_data',
           'rebin_data',
           'load_and_rebin_all_from_nas',
           'save_image')


def load_data_from_nas(nas_path, load_data=True, load_phantom=True, phantom_number=DEFAULT_PHANOM_NUMBER):
    """Load all the needed data from the nas onto your local machine

    This makes loading files much faster.

    Usage on windows where nas is bound to "Z:/"

    python -c "import adutils; adutils.load_data_from_nas('Z:/')"
    """
    phantom_name = phantom_number + 'Phantom_labelled'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    nas_data_path = os.path.join(nas_path, 'CT', 'GPUMCI simulations', '120kV', phantom_name)
    if not os.path.exists(nas_data_path):
        raise IOError('Cannot find NAS data at {}'.format(nas_data_path))

    if load_data:
        glb = glob.glob(os.path.join(nas_data_path, '*.*'))
        for filename in tqdm.tqdm(glb, 'loading data'):
            shutil.copy(filename, data_path)

    nas_phantom_path = os.path.join(nas_path, 'CT', 'GPUMCI simulations', 'code', 'AD_GPUMCI', 'phantoms')
    if not os.path.exists(nas_phantom_path):
        raise IOError('Cannot find NAS data at {}'.format(nas_phantom_path))

    if load_phantom:
        filename = os.path.join(nas_phantom_path,
                                phantom_number + 'Phantom_labelled_no_bed.nii')
        shutil.copy(filename, data_path)


def get_discretization(phantom_number=DEFAULT_PHANOM_NUMBER, use_2D=False):
    # Set geometry for discretization
    pixelSize = 0.44921875
    phantomShape = get_phantom(phantom_number=phantom_number, use_2D=use_2D).shape
    if use_2D:
        volumeSize = np.array([np.round(phantomShape[0]*pixelSize),
                               np.round(phantomShape[1]*pixelSize)])
        volumeOrigin = np.array([-np.round(phantomShape[0]*pixelSize)/2,
                                  -np.round(phantomShape[1]*pixelSize)/2])
        #volumeSize = np.array([230.0, 230.0])
        #volumeOrigin = np.array([-115.0, -115.0])

        # Discretization parameters
        nVoxels = np.array([phantomShape[0],
                            phantomShape[1]])
        #nVoxels = np.array([512, 512])
    else:
        volumeSize = np.array([np.round(phantomShape[0]*pixelSize),
                               np.round(phantomShape[1]*pixelSize),
                               np.round(phantomShape[2]*pixelSize)])
        volumeOrigin = np.array([-np.round(phantomShape[0]*pixelSize)/2,
                                  -np.round(phantomShape[1]*pixelSize)/2,
                                  0])
        #volumeSize = np.array([230.0, 230.0, 141.0546875])
        #volumeOrigin = np.array([-115.0, -115.0, 0])

        # Discretization parameters
        nVoxels = np.array([phantomShape[0],
                            phantomShape[1],
                            phantomShape[2]])
        #nVoxels = np.array([512, 512, 314])

    # Discrete reconstruction space
    reco_space = odl.uniform_discr(volumeOrigin,
                                   volumeOrigin + volumeSize,
                                   nVoxels, dtype='float32')
    return reco_space


def get_ray_trafo(reco_space,
                  use_subset=False,
                  use_rebin=False,
                  rebin_factor=DEFAULT_REBIN_FACTOR,
                  use_window=False,
                  use_2D=False,
                  phantom_number=DEFAULT_PHANOM_NUMBER):

    file_start = 'HelicalSkullCT_' + phantom_number + 'Phantom_no_bed_'

    if use_2D:
        print("Loading geometry")
        geomFile = os.path.join(data_path, (file_start + 'Dose150mGy_2D.geometry.p'))

        with open(geomFile, 'rb') as f:
            if PY3:
                geom = pickle.load(f, encoding='latin1')
            else:
                geom = pickle.load(f)

        A = odl.tomo.RayTransform(reco_space, geom, impl='astra_cuda')

    else:
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
            if use_rebin:
                geomFile = os.path.join(data_path, (file_start + 'Turn_' + str(turn) + '_rebinFactor_' + str(rebin_factor) + '.geometry.p'))
            else:
                geomFile = os.path.join(data_path, (file_start + 'Turn_' + str(turn) + '.geometry.p'))

            # Load pickled geometry (small workaround of incompatibility between Python2/Python3 in pickle)
            with open(geomFile, 'rb') as f:
                if PY3:
                    geom = pickle.load(f, encoding='latin1')
                else:
                    geom = pickle.load(f)

            # X-ray transform
            ray_trafo = odl.tomo.RayTransform(reco_space, geom, impl='astra_cuda')

            if use_window:
                window = odl.tomo.tam_danielson_window(ray_trafo,
                                                       smoothing_width=0.05,
                                                       n_half_rot=3)
                ray_trafo = window * ray_trafo

            Aops.append(ray_trafo)

        A = odl.BroadcastOperator(*Aops)

    return A


def get_fbp(A, use_2D=False):
    if use_2D:
        fbp = odl.tomo.fbp_op(A, padding=True, filter_type='Hamming',
                              frequency_scaling=0.8)
    else:
        # This only works for A a BroadcastOperator, and as long as the
        # Tam-Danielsson window is a vector/matrix and not an operator
        if isinstance(A[0], odl.OperatorLeftVectorMult):
            ops = [Ai.operator for Ai in A]
            vecs = [Ai.vector for Ai in A]
        else:
            ops = A
            vecs = [odl.tomo.tam_danielson_window(Ai, smoothing_width=0.1,
                                                  n_half_rot=3)
                    for Ai in A]

        fbp = odl.ReductionOperator(*[(odl.tomo.fbp_op(
                opi, padding=True, filter_type='Hamming',  # Hann
                frequency_scaling=0.8) * wi) for opi, wi in zip(ops, vecs)])

    return fbp


def get_initial_guess(space):
    """Get initial FBP guess for more refined reconstruction techniques.

    Note: so far this only works with the 70100644Phantom_labelled_no_bed.nii-phantom
    """
    if len(space.shape) == 2: #2D
        arr = np.load(os.path.join(data_path,'reference_reconstruction_{}_{}.npy'.format(space.shape[0], space.shape[1])))
    else:
        arr = np.load(os.path.join(data_path,
                                   'reference_reconstruction_{}_{}_{}.npy'.format(space.shape[0], space.shape[1], space.shape[2])))
    return space.element(arr)


def get_data(A, use_subset=False,
             use_rebin=False,
             rebin_factor=DEFAULT_REBIN_FACTOR,
             use_window=False,
             use_2D=False,
             flip_data=False,
             beam_hardening=True,
             phantom_number=DEFAULT_PHANOM_NUMBER):

    file_start = 'HelicalSkullCT_' + phantom_number + 'Phantom_no_bed_'

    if use_2D:
        print("Loading data")
        dataFile = os.path.join(data_path, (file_start + 'Dose150mGy_2D.data.npy'))
        projections = np.load(dataFile).astype('float32')

        if flip_data:
        # Flip first component of detector due to ODL-fix #1245. Only if using old input data
            projections = projections[:, ::-1]
            #print("Saving file")
            #np.save(dataFile, projections)

        logdata = -np.log(projections / np.max(projections))

        rhs = A.range.element(logdata)

    else:
        # Data
        if use_subset:
            turns = range(13, 16)
        else:
            turns = range(nTurns)

        if not os.path.exists(data_path):
            raise IOError('Could not find files at {}, have you run '
                          'adutils.load_data_from_nas()'.format(data_path))

        imagesTurn = []
        for turn in tqdm.tqdm(turns, 'Loading data'):
            if use_rebin:
                dataFile = os.path.join(data_path, (file_start + 'Dose150mGy_Turn_' + str(turn) + '_rebinFactor_' + str(rebin_factor) + '.data.npy'))
            else:
                dataFile = os.path.join(data_path, (file_start + 'Dose150mGy_Turn_' + str(turn) + '.data.npy'))
            projections = np.load(dataFile).astype('float32')

            if flip_data:
            # Flip first component of detector due to ODL-fix #1245. Only if using old input data
                projections = projections[:, ::-1, :]
                #print("Saving file")
                #np.save(dataFile, projections)

            logdata = -np.log(projections / 6800)

            if beam_hardening:
                logdata = (logdata + 3.0e-2 * logdata ** 2) / (1 + 3e-2)


            if use_window:
                window = odl.tomo.tam_danielson_window(A[turn].operator,  # TODO: ugly
                                                       smoothing_width=0.05,
                                                       n_half_rot=3)
                logdata *= window

            imagesTurn.append(logdata)

        rhs = A.range.element(imagesTurn)

    return rhs


def get_phantom(phantom_number=DEFAULT_PHANOM_NUMBER, use_2D=False, get_Flags=False):
    phantom_name = phantom_number + 'Phantom_labelled_no_bed.nii'
    #nifit data
    #path = '/lcrnas/Reference/CT/GPUMCI simulations/code/AD_GPUMCI/phantoms/'
    phantom = os.path.join(data_path, phantom_name)
    nii = nib.load(phantom)
    label = nii.get_data()
    label = np.asarray(label, dtype=np.float)
    label = np.rot90(label) #Get correct orientation
    label[label == 2] = 5 #Shiftbone
    label[label == 3] = 2 #Shift grey matter
    label[label == 4] = 3 #Shift white matter
    label[label == 5] = 4 #Shift bone

    if not get_Flags:
        if 0:
            # Exact values from spectrum
            label[label == 0] = 0 #Mass attenuation air
            label[label == 1] = 0.019696 #Mass attenuation csf (HU: 15)
            label[label == 2] = 0.020183 #Mass attenuation grey matter (HU: 40)
            label[label == 3] = 0.019884 #Mass attenuation white matter (HU: 25)
            label[label == 4] = 0.04942  #Mass attenuation bone (HU: 1550)
        else:
            # Values from simulation
            label[label == 0] = 0 #Mass attenuation air
            label[label == 1] = 0.021372 #Mass attenuation csf (HU: 15)
            label[label == 2] = 0.021601 #Mass attenuation grey matter (HU: 40)
            label[label == 3] = 0.021451 #Mass attenuation white matter (HU: 25)
            label[label == 4] = 0.048035 #Mass attenuation bone (HU: 1550)


    if use_2D:
        label = label[...,172]

    return label


def plot_data(x, phantom_number=DEFAULT_PHANOM_NUMBER, plot_separately=False, clim = (0.018, 0.022)):

    phantom_name = phantom_number + 'Phantom_no_bed.nii'

    cmap = cm.Greys_r
    x = np.array(x)

    if len(x.shape)<3:
        plt.figure(1)
        plt.title('2D cut')
        plt.imshow(np.flipud(np.transpose(x)), cmap=cmap, clim = clim)
        plt.axis('off')
    else:
        if phantom_name[:8] == '70100644':
            if not plot_separately:
                ax1 = plt.subplot(221)
                ax1.set_title('Coronary cut - anterior segment')
                plt.imshow(np.flipud(np.transpose(x[:,400,:])), cmap=cmap, clim = clim) #257/400/164
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(222)
                ax1.set_title('Coronary cut - caudate')
                plt.imshow(np.flipud(np.transpose(x[:,329,:])), cmap=cmap, clim = clim) #254/329/164
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(223)
                ax1.set_title('Coronary cut - HPC body')
                plt.imshow(np.flipud(np.transpose(x[:,253,:])), cmap=cmap, clim = clim) #257/253/164
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(224)
                ax1.set_title('Coronary cut - HPC tail')
                plt.imshow(np.flipud(np.transpose(x[:,234,:])), cmap=cmap, clim = clim) #257/234/164
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
            else:
                plt.figure(1)
                plt.title('Coronary cut - anterior segment')
                plt.imshow(np.flipud(np.transpose(x[:,400,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(2)
                plt.title('Coronary cut - caudate')
                plt.imshow(np.flipud(np.transpose(x[:,329,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(3)
                plt.title('Coronary cut - HPC body')
                plt.imshow(np.flipud(np.transpose(x[:,253,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(4)
                plt.title('Coronary cut - HPC tail')
                plt.imshow(np.flipud(np.transpose(x[:,234,:])), cmap=cmap, clim = clim)
                plt.axis('off')

            if False: #Additional cuts
                """
                Head of HPC         - x[:,272,:]        #257/272/164
                HPC, difficult part - x[:,295,:]        #205/295/119
                Caudate, difficult  - x[:,299,:]        #257/299/164
                """
        elif phantom_name[:8]  == '70114044':
            if not plot_separately:
                ax1 = plt.subplot(221)
                ax1.set_title('Coronary cut - anterior segment')
                plt.imshow(np.flipud(np.transpose(x[:,489,:])), cmap=cmap, clim = clim) #237/489/169
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(222)
                ax1.set_title('Coronary cut - caudate')
                plt.imshow(np.flipud(np.transpose(x[:,255,:])), cmap=cmap, clim = clim) #237/255/169
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(223)
                ax1.set_title('Coronary cut - HPC body')
                plt.imshow(np.flipud(np.transpose(x[:,358,:])), cmap=cmap, clim = clim) #204/358/121
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(224)
                ax1.set_title('Coronary cut - Amygdala')
                plt.imshow(np.flipud(np.transpose(x[:,305,:])), cmap=cmap, clim = clim) #308/305/120
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
            else:
                plt.figure(1)
                plt.title('Coronary cut - anterior segment')
                plt.imshow(np.flipud(np.transpose(x[:,489,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(2)
                plt.title('Coronary cut - caudate')
                plt.imshow(np.flipud(np.transpose(x[:,255,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(3)
                plt.title('Coronary cut - HPC body')
                plt.imshow(np.flipud(np.transpose(x[:,358,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(4)
                plt.title('Coronary cut - Amygdala')
                plt.imshow(np.flipud(np.transpose(x[:,305,:])), cmap=cmap, clim = clim)
                plt.axis('off')
        elif phantom_name[:8] == '70122044':
            if not plot_separately:
                ax1 = plt.subplot(221)
                ax1.set_title('Coronary cut - anterior segment')
                plt.imshow(np.flipud(np.transpose(x[:,109,:])), cmap=cmap, clim = clim) #378/109/138
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(222)
                ax1.set_title('Coronary cut - caudate')
                plt.imshow(np.flipud(np.transpose(x[:,188,:])), cmap=cmap, clim = clim) #411/188/138
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(223)
                ax1.set_title('Coronary cut - HPC body')
                plt.imshow(np.flipud(np.transpose(x[:,285,:])), cmap=cmap, clim = clim) #384/285/138
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
                ax1 = plt.subplot(224)
                ax1.set_title('Coronary cut - HPC tail')
                plt.imshow(np.flipud(np.transpose(x[:,296,:])), cmap=cmap, clim = clim) #369/296/138
                ax1.axes.xaxis.set_visible(False)
                ax1.axes.yaxis.set_visible(False)
            else:
                plt.figure(1)
                plt.title('Coronary cut - anterior segment')
                plt.imshow(np.flipud(np.transpose(x[:,109,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(2)
                plt.title('Coronary cut - caudate')
                plt.imshow(np.flipud(np.transpose(x[:,188,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(3)
                plt.title('Coronary cut - HPC body')
                plt.imshow(np.flipud(np.transpose(x[:,285,:])), cmap=cmap, clim = clim)
                plt.axis('off')
                plt.figure(4)
                plt.title('Coronary cut - HPC tail')
                plt.imshow(np.flipud(np.transpose(x[:,296,:])), cmap=cmap, clim = clim)
                plt.axis('off')
            if False: #Additional cuts
                """
                Amygdala & HPC      - x[:,248,:]     #397/248/138
                """
        else:
            print('Phantom not recognized. Has it really been simulated?')

def rebin_data(rebin_factor=DEFAULT_REBIN_FACTOR,
               phantom_number=DEFAULT_PHANOM_NUMBER,
               plot_rebin=False):

    file_start = 'HelicalSkullCT_' + phantom_number + 'Phantom_no_bed_'

    if 4000 % rebin_factor:
        raise IOError('Cannot rebin data %i projections with rebin factor %i' %(4000,rebin_factor))

    if not os.path.exists(data_path):
        raise IOError('Could not find files at {}, have you run '
                      'adutils.load_data_from_nas()'.format(data_path))

    for turn in tqdm.tqdm(range(23), "Rebinning data"):
        myFile = os.path.join(data_path,(file_start + 'Dose150mGy_Turn_' + str(turn) + '.data.npy'))
        projections = np.load(myFile)
        size = np.array(np.shape(projections))
        sizeNew = size
        sizeNew[0] /= rebin_factor
        projection_rebin = np.empty(sizeNew)
        for i in range(int(size[0])):
            singleBin = np.mean(projections[i*rebin_factor:(i+1)*rebin_factor,:,:], axis = 0)
            projection_rebin[i,...] = singleBin
        saveName = os.path.join(data_path,(file_start + 'Dose150mGy_Turn_' + str(turn) + '_rebinFactor_'+str(rebin_factor) + '.data.npy'))
        np.save(saveName,projection_rebin)

        geomFile = os.path.join(data_path,(file_start + 'Turn_' + str(turn) + '.geometry.p'))
        with open(geomFile, 'rb') as f:
            if PY3:
                geom = pickle.load(f, encoding='latin1')
            else:
                geom = pickle.load(f)

        nProjection = sizeNew[0]
        angle_partition = odl.uniform_partition(2 * np.pi * turn,
                                                2 * np.pi * (turn + 1),
                                                nProjection)

        geom_rebin = odl.tomo.ConeFlatGeometry(angle_partition,
                                               geom.det_partition,
                                               src_radius=geom.src_radius,
                                               det_radius=geom.det_radius,
                                               pitch=geom.pitch,
                                               offset_along_axis=geom.offset_along_axis)

        pickle.dump(geom_rebin, open(os.path.join(data_path,(file_start + 'Turn_' + str(turn) + '_rebinFactor_' + str(rebin_factor) + '.geometry.p')), 'wb+'))

        if turn == 0:
            projectionsTot = projection_rebin
        else:
            projectionsTot = np.append(projectionsTot,projection_rebin,axis = 0)

    if plot_rebin == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(projectionsTot[:,:,10])
        ax.set_aspect('auto')


def load_and_rebin_all_from_nas(nas_path, rebin_factor=DEFAULT_REBIN_FACTOR, plot_rebin=False):
    for phantom_number in PHANTOM_NUMBERS:
        print('Loading: {}'.format(phantom_number))

        load_data_from_nas(nas_path,
                           load_data=True,
                           load_phantom=True,
                           phantom_number=phantom_number)
        rebin_data(rebin_factor=rebin_factor,
                   phantom_number=phantom_number,
                   plot_rebin=plot_rebin)


def save_image(data, filename, as_nii=True, as_npy=True):
    data = np.asarray(data)
    if as_nii == True:
        new_image = nib.Nifti1Image(data, affine=np.eye(4))
        filename_nii = filename + '.nii'
        nib.save(new_image, filename_nii)
    if as_npy == True:
        filename_npy = filename + '.npy'
        np.save(filename_npy, data)




