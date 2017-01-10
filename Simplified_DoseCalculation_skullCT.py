#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simplified dose calculation of simulated skull CT data. Given for completeness
"""


import numpy as np
import nibabel as nib

def get_spectrum(n):
    # Get spectrum with n points
    path = '/lcrnas/data/Simulated/code/AD_GPUMCI/data/'
    dat = np.loadtxt(path+'spectrumGeriatrics.txt')
    energies = dat[:, 0] / 1000.0
    spectrum = dat[:, 1]

    indices = np.linspace(0, energies.size - 1, n, dtype=int)
    energies = energies[indices]
    spectrum = spectrum[indices]
    spectrum /= spectrum.sum()
    return energies, spectrum

def getPhantom(phantomName):

    #nifit data
    path = '/lcrnas/data/Simulated/code/AD_GPUMCI/phantoms/'
    nii = nib.load(path+phantomName)
    phantomdata = nii.get_data()

    densities = np.zeros_like(phantomdata, dtype=float)
    densities[phantomdata == 1] = 1.015   # fat
    densities[phantomdata == 2] = 1.8   # bone
    densities[phantomdata == 4] = 1.025 # white matter
    densities[phantomdata == 3] = 1.040 # grey matter
    densities = np.reshape(densities, np.shape(phantomdata), order='F')

    # Mat material indices, happen to be very similiar.
    mat = np.zeros(densities.shape, dtype=int, order='F')
    mat[phantomdata == 0] = 0     # air
    mat[phantomdata == 1] = 1     # water
    mat[phantomdata == 2] = 2     # bone
    mat[phantomdata == 4] = 4     # white matter
    mat[phantomdata == 3] = 3     # grey matter

    return densities, mat

energies, spectrum = get_spectrum(10)
photons_per_pixel_one_run = 1000
simNum = 120
photons_per_pixel = photons_per_pixel_one_run*simNum
#spectrum *= photons_per_pixel
numberOfPixels = 500*20
numberOfTurns = 23
numberOfProjections = 4000

energyInPerProjection = photons_per_pixel* numberOfPixels * np.sum(energies*spectrum)

energyHeadPerProjection = []
for turn in range(numberOfTurns):
    print(turn)
    file_name = ('/lcrnas/data//Simulated/120kV/raw/' +
                 'helical_proj_70100644Phantom_labelled_no_bed_' + str(simNum) +
                 '_Simulations_Turn_num_' + str(turn) + '.npy')
    proj = np.load(file_name)
    for projection in range(numberOfProjections):
        energyHeadPerProjection.append(energyInPerProjection -
                                       np.sum(proj[projection,...]))

MeV2J = 1.6021773e-13
energyHeadTot = np.sum(energyHeadPerProjection) *MeV2J

phantomName = '70100644Phantom_labelled_no_bed.nii'
voxelSize = ((230./512)/10)**3
den, mat = getPhantom(phantomName)
massHead = np.sum(den*voxelSize)/1000

#massHead = 4

dose = energyHeadTot/massHead
dose *= 1000

print('Dose is: %f mGy' % dose)

