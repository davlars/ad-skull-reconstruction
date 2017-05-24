"""
TGV reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
import adutils
import scipy


def optimal_parameters(reconstruction, fom, phantoms, data,
                       initial_param=0):
    """Find the optimal parameters for a reconstruction method.

    Notes
    -----
    For a forward operator :math:`A : X \to Y`, a reconstruction operator
     parametrized by :math:`\theta` is some operator :math:`R_\theta : Y \to X`
     such that

    .. math::
        R_\theta(A(x)) \approx x.

    The optimal choice of :math:`\theta` is given by

    .. math::
        \theta = \argmin_\theta fom(R(A(x) + noise), x)

    where :math:`fom : X \times X \to \mathbb{R}` is a Figure of Merit.


    Parameters
    ----------
    reconstruction : callable
        Function that takes two parameters:

            * data : The data to be reconstructed
            * parameters : Parameters of the reconstruction method

        The function should return the reconstructed image.
    fom : callable
        Function that takes two parameters:

            * reconstructed_image
            * true_image

        and returns a scalar Figure of Merit.
    phantoms : sequence
        True images.
    data : sequence
        The data to be reconstructed.
    initial_param : array-like
        Initial guess for the parameters.
    """

    def func(lam):
        # Function to be minimized by scipy
        return sum(fom(reconstruction(datai, lam), phantomi)
                   for phantomi, datai in zip(phantoms, data))

    # Pick resolution to fit the one used by the space
    tol = np.finfo(phantoms[0].space.dtype).resolution * 10

    initial_param = np.asarray(initial_param)

    if initial_param.size == 1:
        bracket = [initial_param - tol, initial_param + tol]
        result = scipy.optimize.minimize_scalar(func,
                                                bracket=bracket,
                                                tol=tol,
                                                bounds=None,
                                                options={'disp': False})
        return result.x
    else:
        # Use a gradient free method to find the best parameters
        parameters = scipy.optimize.fmin_powell(func, initial_param,
                                                xtol=tol,
                                                ftol=tol,
                                                disp=False)
        return parameters


# Discretization
space = adutils.get_discretization(use_2D=True)

# Forward operator (in the form of a broadcast operator)
ray_trafo = adutils.get_ray_trafo(space, use_2D=True)

# Data
data = adutils.get_data(ray_trafo, use_2D=True)
phantom = adutils.get_phantom(use_2D=True)

fbp_op = adutils.get_fbp(ray_trafo, use_2D=True)
fbp = fbp_op(data)


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(space, method='forward')

gradient_back = odl.Gradient(space, method='backward')
eps = odl.DiagonalOperator(gradient_back, space.ndim)

# Create the domain of the problem, given by the reconstruction space and the
# range of the gradient on the reconstruction space.
domain = odl.ProductSpace(space, gradient.range)

# Column vector of three operators defined as:
# 1. Computes ``A(x)``
# 2. Computes ``grad(x) - y``
# 3. Computes ``eps(y)``
op = odl.BroadcastOperator(
    ray_trafo * odl.ComponentProjection(domain, 0),
    odl.ReductionOperator(gradient, odl.ScalingOperator(gradient.range, -1)),
    eps * odl.ComponentProjection(domain, 1))

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching

op_norm = 1.1 * odl.power_method_opnorm(op)


def reconstruction(proj_data, parameters):
    lam1, lam2 = parameters

    print('lam1 = {}, lam2 = {}'.format(lam1, lam2))

    if lam1 < 0 or lam2 < 0:
        return np.inf * space.one()

    # The l1-norms scaled by regularization paramters
    l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(proj_data)
    l1_norm_1 = lam1 * odl.solvers.L1Norm(gradient.range)
    l1_norm_2 = lam2 * odl.solvers.L1Norm(eps.range)

    # Combine functionals, order must correspond to the operator K
    f = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)

    # --- Select solver parameters and solve using Chambolle-Pock --- #

    # Estimated operator norm, add 10 percent to ensure
    # ||K||_2^2 * sigma * tau < 1

    niter = 400  # Number of iterations
    tau = 1.0 / op_norm  # Step size for the primal variable
    sigma = 1.0 / op_norm  # Step size for the dual variable
    gamma = 0.5

    # Choose a starting point
    x0 = fbp.copy()
    x1 = gradient.range.zero()

    x = op.domain.element([x0, x1])

    # Run the algorithm
    odl.solvers.chambolle_pock_solver(
        x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma)

    return x[0]

initial_param = [1e-3, 1e-4]

phantoms = [space.element(phantom)]
datas = [data]


def fom(x1, x2):
    result = space.dist(x1, x2)
    print('FOM: {}'.format(result))
    return result


optimal_parameters = optimal_parameters(reconstruction,  fom,
                                        phantoms, datas,
                                        initial_param=initial_param)
