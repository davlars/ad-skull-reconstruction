"""
FBP reconstruction example for simulated Skull CT data
"""

from odl.contrib import fom
import numpy as np
import adutils

for phantom_number in adutils.PHANTOM_NUMBERS:
    # Discretization
    reco_space = adutils.get_discretization(phantom_number=phantom_number)

    # Forward operator (in the form of a broadcast operator)
    A = adutils.get_ray_trafo(reco_space,
                              phantom_number=phantom_number,
                              use_rebin=True)

    # Define fbp
    fbp = adutils.get_fbp(A)

    # Data
    rhs = adutils.get_data(A,
                           phantom_number=phantom_number,
                           use_rebin=True,
                           beam_hardening=True)

    # Reconstruct
    x = fbp(rhs)

    # Reference
    phantom = reco_space.element(adutils.get_phantom(phantom_number=phantom_number))

    if True:
        x.show(coords=[None, 0, None], clim=(0.021, 0.022))
        x.show(coords=[0, None, None], clim=(0.021, 0.022))
        x.show(coords=[None, None, 90], clim=(0.021, 0.022))

        x.show(coords=[0, None, 90])

        diff = (x - phantom)
        clim = [-0.002, 0.002]

        diff.show(coords=[None, 0, None], clim=clim)
        diff.show(coords=[0, None, None], clim=clim)
        diff.show(coords=[None, None, 90], clim=clim)

    print('phantom: {}, psnr: {}'.format(phantom_number, fom.psnr(x, phantom)))

    if 1:
        # Save
        x_arr = x.asarray()
        p_arr = phantom.asarray()
        for slc in range(phantom.shape[-1]):
            name = 'data/learning/{}_{}'.format(phantom_number, slc)
            np.savez(name, fbp=x_arr[..., slc], phantom=p_arr[..., slc])
