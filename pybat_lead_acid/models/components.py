"""
The components that make up the model.
"""
import numpy as np

def simple_diffusion(c, grad, div, flux_bc_left, flux_bc_right, source=0):
    """The 1D diffusion equation.

    Parameters
    ----------
    c : array_like, shape (n,)
        The quantity being diffused.
    grad : function
        The gradient operator.
    div : function
        The divergence operator.
    flux_bc_left : array_like, shape (1,)
        Flux on the left-hand side (Neumann BC).
    flux_bc_right : array_like, shape (1,)
        Flux on the right-hand side (Neumann BC).
    source : float or array_like, shape (n,), optional
        Source term in the PDE.

    Returns
    -------
    dydt : array_like, shape (n,)
        The time derivative.

    """
    # Calculate internal flux
    N_internal = - grad(c)

    # Add boundary conditions (Neumann)
    N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

    # Calculate time derivative
    dcdt = - div(N) + source

    return dcdt
