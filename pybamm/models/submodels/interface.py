#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


def homogeneous_reaction():
    """
    Homogeneous reaction at the electrode-electrolyte interface
    """

    current_neg = (
        pybamm.Scalar(1, domain=["negative electrode"]) / pybamm.standard_parameters.ln
    )
    current_sep = pybamm.Scalar(0, domain=["separator"])
    current_pos = (
        -pybamm.Scalar(1, domain=["positive electrode"]) / pybamm.standard_parameters.lp
    )
    return pybamm.Concatenation(current_neg, current_sep, current_pos)


def butler_volmer_lead_acid(c, phi, domain=None):
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
        # raise error if no domain can be found
        if c.domain == [] and phi.domain == []:
            raise ValueError(
                "domain cannot be None if c.domain and phi.domain are empty"
            )
        # otherwise read domain from c and phi, making sure they are consistent with
        # each other
        else:
            if phi.domain == c.domain or c.domain == [] or phi.domain == []:
                domain = c.domain
            else:
                raise pybamm.DomainError(
                    "c and phi must have the same (or empty) domain"
                )

    # Get the current densities based on domain
    if domain == ["negative electrode"]:
        j0n = exchange_current_density(c, ["negative electrode"])
        etan = phi - pybamm.standard_parameters_lead_acid.U_Pb(c)
        return j0n * etan  # Function(etan, np.sinh)
    elif domain == ["positive electrode"]:
        j0p = exchange_current_density(c, ["positive electrode"])
        etap = phi - pybamm.standard_parameters_lead_acid.U_PbO2(c)
        return j0p * etap  # Function(etap, np.sinh)
    # To get current density across the whole domain, unpack and call this function
    # again in the subdomains, then concatenate
    elif domain == ["negative electrode", "separator", "positive electrode"]:
        # Unpack c
        if all([isinstance(var, pybamm.Concatenation) for var in [c, phi]]):
            cn, cs, cp = c.orphans
            phin, phis, phip = phi.orphans
        else:
            raise ValueError(
                "c and phi must both be Concatenations, not '{}' and '{}'".format(
                    type(c), type(phi)
                )
            )
        # Negative electrode
        current_neg = butler_volmer_lead_acid(cn, phin, domain=["negative electrode"])
        # Separator
        current_sep = pybamm.Scalar(0, domain=["separator"])
        # Positive electrode
        current_pos = butler_volmer_lead_acid(cp, phip, domain=["positive electrode"])
        # Concatenate
        return pybamm.Concatenation(current_neg, current_sep, current_pos)
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))


def exchange_current_density(c, domain=None):
    """The exchange current-density as a function of concentration

    Parameters
    ----------
    c : :class:`pybamm.Variable`
        A Variable representing the concentration
    domain : string
        Which domain to calculate the exchange current density in ("negative electrode"
        or "positive electrode"). Default is None, in which case the domain is c.domain

    Returns
    -------
    :class:`pybamm.Symbol`
        The exchange-current density
    """
    if domain is None:
        # read domain from c if it exists
        if c.domain != []:
            domain = c.domain
        # otherwise raise error
        else:
            raise ValueError("domain cannot be None if c.domain is empty")

    if domain == ["negative electrode"]:
        # concentration domain should be empty or the same as domain
        if c.domain not in [["negative electrode"], []]:
            raise pybamm.DomainError("""concentration and domain do not match""")
        iota_ref_n = pybamm.standard_parameters_lead_acid.iota_ref_n
        return iota_ref_n * c
    elif domain == ["positive electrode"]:
        # concentration domain should be empty or the same as domain
        if c.domain not in [["positive electrode"], []]:
            raise pybamm.DomainError("""concentration and domain do not match""")
        iota_ref_p = pybamm.standard_parameters_lead_acid.iota_ref_p
        Ve = pybamm.standard_parameters_lead_acid.Ve
        Vw = pybamm.standard_parameters_lead_acid.Vw
        cw = (1 - c * Ve) / Vw
        return iota_ref_p * c ** 2 * cw
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))
