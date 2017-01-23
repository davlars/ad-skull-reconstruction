"""
Wavelet reconstruction example for simulated Skull CT data - parameter search
"""
import odl
import numpy as np
import adutils

# Discretization
reco_space = adutils.get_discretization()

# Forward operator (in the form of a broadcast operator)
A = adutils.get_ray_trafo(reco_space, use_rebin=True)

# Data
rhs = adutils.get_data(A, use_rebin=True)

# Create wavelet operator
W = odl.trafos.WaveletTransform(reco_space, wavelet='haar', nlevels=5)

# The wavelets bases are normalized to constant norm regardless of scale.
# since we want to penalize "small" wavelets more than "large" ones, we need
# to weight by the scale of the wavelets.
# The "area" of the wavelets scales as 2 ^ scale, but we use a slightly smaller
# number in order to allow some high frequencies.
scales = W.scales()

for power in [2.2, 2.5, 2.8]:
    WtrafoScaled = np.power(power, scales) * W

    # Column vector of operators
    op = odl.BroadcastOperator(A, WtrafoScaled)

    Anorm = odl.power_method_opnorm(A[1], maxiter=2)
    Dnorm = odl.power_method_opnorm(WtrafoScaled,
                                    xstart=odl.phantom.white_noise(W.domain),
                                    maxiter=10)

    # Estimated operator norm, add 10 percent
    op_norm = 1.1 * np.sqrt(len(A.operators)*(Anorm**2) + Dnorm**2)

    print('Norm of the product space operator: {}'.format(op_norm))

    lambs = (1e-7, 2e-7, 4e-7, 8e-7, 16e-7, 32e-7, 64e-7, 128e-7)

    # Use FBP as initial guess
    x_init = adutils.get_initial_guess(reco_space)

    for lamb in lambs:
        print('Running power={}, lambda={}'.format(power, lamb))

        # l2-squared data matching
        l2_norm = odl.solvers.L2NormSquared(A.range).translated(rhs)

        # Isotropic TV-regularization i.e. the l1-norm
        l1_norm = lamb * odl.solvers.L1Norm(W.range)

        # Combine functionals
        f = odl.solvers.SeparableSum(l2_norm, l1_norm)

        # Set g functional to positivity constraint
        g = odl.solvers.IndicatorNonnegativity(op.domain)

        # Acceleration parameter
        gamma = 0.4

        # Step size for the proximal operator for the primal variable x
        tau = 10.0 / op_norm
        sigma = 1.0 / (op_norm ** 2 * tau)

        title = 'wavelet reco {}'.format(lamb)

        pth = 'data/results/wavelet/power_{}_lambda_{}_{{:04d}}.png'.format(power, '{:8.8f}'.format(float(lamb)))

        # Reconstruct
        callback = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow(title, coords=[None, 0, None], clim=[0.018, 0.022]) &
                    odl.solvers.CallbackShow(title, coords=[0, None, None], clim=[0.018, 0.022]) &
                    odl.solvers.CallbackShow(title, coords=[None, None, 60], clim=[0.018, 0.022],
                                             saveto=pth))

        # Use the FBP as initial guess
        x = x_init.copy()

        niter = 200
        odl.solvers.chambolle_pock_solver(x, f, g, op, tau=tau, sigma=sigma,
                                          niter=niter, gamma=gamma, callback=callback)
