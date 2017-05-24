"""
TGV reconstruction example for simulated Skull CT data
"""

import odl
import numpy as np
import adutils

# Discretization
space = adutils.get_discretization(use_2D=True)

# Forward operator (in the form of a broadcast operator)
ray_trafo = adutils.get_ray_trafo(space, use_2D=True)

# Data
data = adutils.get_data(ray_trafo, use_2D=True)
phantom = space.element(adutils.get_phantom(use_2D=True))

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
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# The l1-norms scaled by regularization paramters
lam1 = 0.1
lam2 = 1.0 * lam1
l1_norm_1 = lam1 * odl.solvers.L1Norm(gradient.range)
l1_norm_2 = lam2 * odl.solvers.L1Norm(eps.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 400  # Number of iterations
tau = 0.1  # Step size for the primal variable
sigma = 1.0 / (tau * op_norm ** 2)  # Step size for the dual variable
gamma = 0.1

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow(clim=[0.018, 0.022], indices=0, step=10) &
            odl.solvers.CallbackShow(indices=1, step=10))

# Choose a starting point
x0 = fbp.copy()
x1 = gradient.range.zero() # (x0)

x = op.domain.element([x0, x1]).copy()

callback &= lambda x: print(space.dist(phantom, x[0]))

# Run the algorithm
odl.solvers.chambolle_pock_solver(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
    callback=callback)

# Display images
x[0].show(clim=[0.018, 0.022], title='TGV reconstruction')
x[1].show(title='Derivatives', force_show=True)