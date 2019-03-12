#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np


def homogeneous_reaction(domain):
    """
    Homogeneous reaction at the electrode-electrolyte interface

    Parameters
    ----------
    domain : iterable of strings
        If feed in just a single domain then will return a Scalar which will
        remain as a scalar after discretisation

        If feed in ["negative electrode", "separator", "positive electrode"]
        will return a concatenation of scalars. Concatenations
        will be processed into a vector upon discretisation.

    """
    current = pybamm.standard_parameters.current_with_time

    if domain == ["negative electrode"]:
        exchange_current = current / pybamm.standard_parameters.l_n
    elif domain == ["separator"]:
        exchange_current = pybamm.Scalar(0)
    elif domain == ["positive electrode"]:
        exchange_current = -current / pybamm.standard_parameters.l_p

    elif domain == ["negative electrode", "separator", "positive electrode"]:
        return pybamm.Concatenation(
            *[pybamm.Broadcast(homogeneous_reaction([dom]), [dom]) for dom in domain]
        )
    else:
        raise pybamm.DomainError("{} is not a valid domain".format(domain))

    # set the domain (required for processing boundary conditions)
    exchange_current.domain = domain

    return exchange_current


def exchange_current_density(c_e, ck_surf=None, domain=None):
    """The exchange current-density as a function of concentration

    Parameters
    ----------
    c_e : :class:`pybamm.Variable`
        The electrolyte concentration
    ck_surf : :class:`pybamm.Variable`
        The concentration of lithium on the surface of a particle
    domain : string
        Which domain to calculate the exchange current density in ("negative electrode"
        or "positive electrode"). Default is None, in which case the domain is\
        c_e.domain

    Returns
    -------
    :class:`pybamm.Symbol`
        The exchange-current density
    """
    if domain is None:
        # read domain from c if it exists
        if c_e.domain != []:
            domain = c_e.domain
        # otherwise raise error
        else:
            raise ValueError("domain cannot be None if c_e.domain is empty")

    sp = pybamm.standard_parameters
    m = {("negative electrode",): sp.m_n, ("positive electrode",): sp.m_p}[
        tuple(domain)
    ]
    if ck_surf:
        return m * c_e ** (1 / 2) * ck_surf ** (1 / 2) * (1 - ck_surf) ** (1 / 2)
    else:
        if domain == ["negative electrode"]:
            return sp.m_n * c_e
        elif domain == ["positive electrode"]:
            c_w = (1 - c_e * sp.V_e) / sp.V_w
            return sp.m_p * c_e ** 2 * c_w


def butler_volmer(param, c_e, Delta_phi, ck_surf=None, domain=None):
    """
    Butler-Volmer reactions for lead-acid chemistry

    .. math::
        j = j_0(c) * \\sinh(\\phi - U(c)),

        \\text{where} \\phi = \\Phi_\\text{s} - \\Phi

    Parameters
    ----------
    c : :class:`pybamm.Symbol`
        The electrolyte concentration
    phi : :class:`pybamm.Symbol`
        The difference betweent the solid potential and electrolyte potential
    domain : iterable of strings
        The domain in which to calculate the interfacial current density. Default is
        None, in which case the domain is calculated based on c and phi or defaults to
        the domain spanning the whole cell

    Returns
    -------
    :class:`pybamm.Symbol`
        The interfacial current density in the appropriate domain
    """
    if domain is None:
        # check and get domain using the functionality from Binary Operator
        domain = pybamm.BinaryOperator("", c_e, Delta_phi).domain
        # raise error if no domain can be found
        if domain == []:
            raise ValueError(
                "domain cannot be None if c_e.domain and Delta_phi.domain are empty"
            )

    # Get the concentration for ocp (either surface concentration, if given, or
    # electrolyte concentration otherwise)
    if ck_surf:
        c_ocp = ck_surf
    else:
        c_ocp = c_e

    # Get the current densities based on domain
    if domain == ["negative electrode"]:
        j0_n = exchange_current_density(c_e, ck_surf, ["negative electrode"])
        eta_n = Delta_phi - param.U_n(c_ocp)
        return j0_n * pybamm.Function(np.sinh, eta_n)
    elif domain == ["positive electrode"]:
        j0_p = exchange_current_density(c_e, ck_surf, ["positive electrode"])
        eta_p = Delta_phi - param.U_p(c_ocp)
        return j0_p * pybamm.Function(np.sinh, eta_p)
    # To get current density across the whole domain, unpack and call this function
    # again in the subdomains, then concatenate
    elif domain == ["negative electrode", "separator", "positive electrode"]:
        # Unpack c
        if all([isinstance(var, pybamm.Concatenation) for var in [c_e, Delta_phi]]):
            c_e_n, c_e_s, c_e_p = c_e.orphans
            Delta_phi_n, Delta_phi_s, Delta_phi_p = Delta_phi.orphans
        else:
            raise TypeError(
                """
                c_e and Delta_phi must both be Concatenations, not '{}' and '{}'
                """.format(
                    type(c_e), type(Delta_phi)
                )
            )
        if ck_surf:
            if isinstance(ck_surf, pybamm.Concatenation):
                cn_surf, cp_surf = ck_surf.orphans
            else:
                raise TypeError(
                    "ck_surf must be a Concatenation, not '{}'".format(type(ck_surf))
                )
        else:
            cn_surf, cp_surf = None, None
        # Negative electrode
        current_neg = butler_volmer(
            param, c_e_n, Delta_phi_n, cn_surf, ["negative electrode"]
        )
        # Separator
        current_sep = pybamm.Broadcast(0, ["separator"])
        # Positive electrode
        current_pos = butler_volmer(
            param, c_e_p, Delta_phi_p, cp_surf, ["positive electrode"]
        )
        # Concatenate
        return pybamm.Concatenation(current_neg, current_sep, current_pos)
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))
