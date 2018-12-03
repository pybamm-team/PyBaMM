#
# The components that make up the models.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np


def electrolyte_diffusion(param, c, operators, flux_bcs, j):
    """The 1D diffusion equation.

    Parameters
    ----------
    param : pybamm.parameters.Parameter() instance
        The parameters of the simulation
    c : array_like, shape (n,)
        The electrolyte concentration.
    operators : pybamm.operators.Operators() instance
        The spatial operators.
    flux_bcs : 2-tuple of array_like, shape (1,)
        Flux at the boundaries (Neumann BC).
    j : array_like, shape (n,)
        The interfacial current density.

    Returns
    -------
    dcdt : array_like, shape (n,)
        The time derivative.

    """
    # Calculate internal flux
    N_internal = -operators.grad(c)

    # Add boundary conditions (Neumann)
    flux_bc_left, flux_bc_right = flux_bcs
    N = np.concatenate([flux_bc_left, N_internal, flux_bc_right])

    # Calculate time derivative
    dcdt = -operators.div(N) + param.s * j

    return dcdt


def electrolyte_current(param, variables, operators, current_bcs, j):
    """The 1D diffusion equation.

    Parameters
    ----------
    param : pybamm.parameters.Parameter() instance
        The parameters of the simulation
    variables : 2-tuple (c, e) of array_like, shape (n,)
        The concentration, and potential difference.
    operators : pybamm.operators.Operators() instance
        The spatial operators.
    current_bcs : 2-tuple of array_like, shape (1,)
        Flux at the boundaries (Neumann BC).
    j : array_like, shape (n,)
        Interfacial current density.

    Returns
    -------
    dedt : array_like, shape (n,)
        The time derivative of the potential.

    """
    # Calculate current density
    i = -current(variables, operators, current_bcs)

    # Calculate time derivative
    dedt = 1 / param.gamma_dl * (operators.div(i) - j)

    return dedt


def current(param, variables, operators, current_bcs):
    """The 1D current.

    Parameters
    ----------
    param : pybamm.parameters.Parameter() instance
        The parameters of the simulation
    variables : 2-tuple (c, e) of array_like, shape (n,)
        The concentration, and potential difference.
    operators : pybamm.operators.Operators() instance
        The spatial operators.
    current_bcs : 2-tuple of array_like, shape (1,)
        Flux at the boundaries (Neumann BC).

    Returns
    -------
    i : array_like, shape (n+1,)
        The current density.

    """
    c, e = variables

    kappa_over_c = 1
    kappa = 1

    # Calculate inner current
    i_inner = kappa_over_c * operators.grad(c) + kappa * operators.grad(e)

    # Add boundary conditions
    lbc, rbc = current_bcs
    i = np.concatenate([lbc, i_inner, rbc])

    return i


def butler_volmer(param, cn, cs, cp, en, ep):
    """Calculates the interfacial current densities
    using Butler-Volmer kinetics.

    Parameters
    ----------
    param : :class:`pybamm.parameters.Parameters` instance
        The parameters of the simulation.
    cn : array_like, shape (n,)
        The electrolyte concentration in the negative electrode.
    cs : array_like, shape (s,)
        The electrolyte concentration in the separator.
    cp : array_like, shape (p,)
        The electrolyte concentration in the positive electrode.
    en : array_like, shape (n,)
        The potential difference in the negative electrode.
    ep : array_like, shape (p,)
        The potential difference in the positive electrode.

    Returns
    -------
    j : array_like, shape (n+s+p,)
        The interfacial current density across the whole cell.
    jn : array_like, shape (n,)
        The interfacial current density in the negative electrode.
    jp : array_like, shape (p,)
        The interfacial current density in the positive electrode.

    """
    jn = param.iota_ref_n * cn * np.sinh(en - param.U_Pb(cn))
    js = 0 * cs
    jp = param.iota_ref_p * cp ** 2 * param.cw(cp) * np.sinh(ep - param.U_PbO2(cp))

    j = np.concatenate([jn, js, jp])
    return j, jn, jp
