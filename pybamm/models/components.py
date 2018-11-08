"""
The components that make up the model.
"""
import numpy as np

def simple_diffusion(c, operators, flux_bcs, source=0):
    """The 1D diffusion equation.

    Parameters
    ----------
    c : array_like, shape (n,)
        The quantity being diffused.
    operators : pybamm.operators.Operators() instance
        The spatial operators.
    flux_bc_left : 2-tuple of array_like, shape (1,)
        Flux at the boundaries (Neumann BC).
    source : int or float or array_like, shape (n,), optional
        Source term in the PDE.

    Returns
    -------
    dydt : array_like, shape (n,)
        The time derivative.

    """
    # Calculate internal flux
    N_internal = - operators.grad_x(c)

    # Add boundary conditions (Neumann)
    flux_bc_left, flux_bc_right = flux_bcs
    N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

    # Calculate time derivative
    dcdt = - operators.div_x(N) + source

    return dcdt

def butler_volmer(param, cn, cs, cp, en, ep):
    jn = param.iota_ref_n * cn * np.sinh(en - param.U_Pb(cn))
    js = 0*cs
    jp = (param.iota_ref_p * cp**2 * param.cw(cp)
          * np.sinh(ep - param.U_PbO2(cp)))

    j = np.concatenate([jn, js, jp])
    return j, jn, jp
