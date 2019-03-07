#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np


def homogeneous_reaction(domain):
    """ Homogeneous reaction at the electrode-electrolyte interface """

    # If feed in just a single domain then will return a Scalar which will
    # remain as a scalar after discretisation

    # If feed in ["negative electrode", "separator", "positive electrode"]
    # will return a concatenation of scalars. Concatenations
    # will be processed into a vector upon discretisation.
    current = pybamm.standard_parameters.current_with_time

    if domain == ["negative electrode"]:
        exchange_current = current / pybamm.standard_parameters.l_n
    elif domain == ["separator"]:
        exchange_current = pybamm.Scalar(0)
    elif domain == ["positive electrode"]:
        exchange_current = -current / pybamm.standard_parameters.l_p

    elif domain == ["negative electrode", "separator", "positive electrode"]:
        # hack to make the concatenation work. Concatenation needs some work
        current_neg = pybamm.Broadcast(
            current / pybamm.standard_parameters.l_n, ["negative electrode"]
        )
        current_pos = pybamm.Broadcast(
            -current / pybamm.standard_parameters.l_p, ["positive electrode"]
        )
        return pybamm.Concatenation(
            current_neg, pybamm.Broadcast(0, ["separator"]), current_pos
        )
    else:
        raise NotImplementedError("{} is not a valid domain".format(domain))

    # set the domain (required for processing boundary conditions)
    exchange_current.domain = domain

    return exchange_current


def butler_volmer(m, U_eq, c_e, Delta_phi, ck_surf=None, domain=None):
    """
    Butler-Volmer reactions.

    .. math::
        j = j_0(c_e, c_\\text{k}) * \\sinh(\\Delta\\phi - U_\\text{eq}(c_\\text{k})),

        \\text{where} \\Delta \\phi = \\Phi_\\text{k} - \\Phi

    Parameters
    ----------
    m: :class:`pybamm.Parameter`
        The dimensionless reaction rate constant (dimensionless reference
        exchange current density)
    U_eq: :class: `pyabmm.Parameter`
        The open circuit potential
    c_e : :class:`pybamm.Symbol`
        The electrolyte concentration
    Delta_phi : :class:`pybamm.Symbol`
        The difference between the electrode potential and the electrolyte potential.
    ck_surf: :class: `pybamm.Variable`
        The concentration of lithium on the surface of a particle.
    domain : iterable of strings
        The domain in which to calculate the interfacial current density. Default is
        None, in which case the domain is calculated based on c and phi or defaults to
        the domain spanning the whole cell

    Returns
    -------
    :class:`pybamm.Symbol`
        The dimensionless interfacial current density (=dimensionless flux density)
    """
    if domain is None:
        # raise error if no domain can be found
        if c_e.domain == [] and Delta_phi.domain == []:
            raise ValueError(
                "domain cannot be None if c_e.domain and phi_e.domain are empty"
            )
        # otherwise read domain from c and phi, making sure they are consistent with
        # each other
        else:
            if (
                Delta_phi.domain == c_e.domain
                or c_e.domain == []
                or Delta_phi.domain == []
            ):
                domain = c_e.domain
            else:
                raise pybamm.DomainError(
                    "c_e and phi_e must have the same (or empty) domain"
                )

    # Get the current densities based on domain
    if domain == ["negative electrode"]:
        j0n = exchange_current_density(
            m, c_e, ck_surf=ck_surf, domain=["negative electrode"]
        )
        eta_n = Delta_phi - U_eq(ck_surf)
        return j0n * eta_n  # Function(etan, np.sinh)
    elif domain == ["positive electrode"]:
        j0p = exchange_current_density(
            m, c_e, ck_surf=ck_surf, domain=["positive electrode"]
        )
        eta_p = Delta_phi - U_eq(ck_surf)
        return j0p * eta_p  # Function(etap, np.sinh)
    # To get current density across the whole domain, unpack and call this function
    # again in the subdomains, then concatenate
    elif domain == ["negative electrode", "separator", "positive electrode"]:

        # Unpack c
        if ck_surf is None:
            variables = [c_e, Delta_phi, U_eq]
        else:
            variables = [c_e, Delta_phi, U_eq, ck_surf]

        if all([isinstance(var, pybamm.Concatenation) for var in variables]):
            c_en, c_es, c_ep = c_e.orphans
            Delta_phi_n, Delta_phi_s, Delta_phi_p = Delta_phi.orphans
            U_n, U_s, U_p = U_eq.orphans
            m_n, m_s, m_p = m.orphans
        else:
            raise ValueError(
                "c_e, Delta_phi, U_eq, (and ck_surf)\
                must both be Concatenations, not '{}' and '{}', '{}".format(
                    type(c_e), type(Delta_phi), type(U_eq)
                )
            )
        # Negative electrode
        j_n = butler_volmer(
            m, U_n, c_en, Delta_phi_n, ck_surf=ck_surf, domain=["negative electrode"]
        )
        # Separator
        j_s = pybamm.Scalar(0, domain=["separator"])
        # Positive electrode
        j_p = butler_volmer(
            m, U_p, c_e, Delta_phi, ck_surf=ck_surf, domain=["positive electrode"]
        )
        # Concatenate
        return pybamm.Concatenation(j_n, j_s, j_p)
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))


def exchange_current_density(m, c_e, ck_surf=None, domain=None):
    """The exchange current-density as a function of concentration

    Parameters
    ----------
    m: :class: `pybamm.Parameter`
        The dimensionless reaction rate constant (dimensionless reference exchange
        current density)
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

    if domain[0] not in pybamm.KNOWN_DOMAINS:
        raise pybamm.DomainError("{} is not in known domains".format(domain))

    if ck_surf is not None:
        # check that ck_surf and c_e have are both negative or positive
        if (ck_surf.domain == ["negative particle"]) and (
            domain != ["negative electrode"]
        ):
            raise ValueError(
                "ck_surf and c_e must both be on respective 'negative' or 'positive'\
                        domains"
            )
        if (ck_surf.domain == ["positive particle"]) and (
            domain != ["positive electrode"]
        ):
            raise ValueError(
                "ck_surf and c_e must both be on respective 'negative' or 'positive'\
                        domains"
            )

    # need to check that raises assert when c_e.domain != domain
    if domain == ["negative electrode"]:
        if c_e.domain not in [["negative electrode"], []]:
            raise pybamm.DomainError("""concentration and domain do not match""")
    elif domain == ["positive electrode"]:
        if c_e.domain not in [["positive electrode"], []]:
            raise pybamm.DomainError("""concentration and domain do not match""")

    # I actually don't like the set of if statements here.
    if ck_surf is not None:
        # only activated by li-ion
        return m * c_e ** (1 / 2) * ck_surf ** (1 / 2) * (1 - ck_surf) ** (1 / 2)
    elif domain == ["negative electrode"]:
        # only activated in neg of lead-acid
        return m * c_e
    elif domain == ["positive electrode"]:
        # only activated in pos of lead-acid
        V_e = pybamm.standard_parameters.V_e
        V_w = pybamm.standard_parameters.V_w
        c_w = (1 - c_e * V_e) / V_w
        return m * c_e ** 2 * c_w
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))


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
        j0_n = exchange_current_density_lead_acid(c, ["negative electrode"])
        eta_n = phi - pybamm.standard_parameters_lead_acid.U_n(c)
        return j0_n * pybamm.Function(np.sinh, eta_n)
    elif domain == ["positive electrode"]:
        j0_p = exchange_current_density_lead_acid(c, ["positive electrode"])
        eta_p = phi - pybamm.standard_parameters_lead_acid.U_p(c)
        return j0_p * pybamm.Function(np.sinh, eta_p)
    # To get current density across the whole domain, unpack and call this function
    # again in the subdomains, then concatenate
    elif domain == ["negative electrode", "separator", "positive electrode"]:
        # Unpack c
        if all([isinstance(var, pybamm.Concatenation) for var in [c, phi]]):
            c_n, c_s, c_p = c.orphans
            phi_n, phi_s, phi_p = phi.orphans
        else:
            raise ValueError(
                "c and phi must both be Concatenations, not '{}' and '{}'".format(
                    type(c), type(phi)
                )
            )
        # Negative electrode
        current_neg = butler_volmer_lead_acid(c_n, phi_n, domain=["negative electrode"])
        # Separator
        current_sep = pybamm.Broadcast(0, ["separator"])
        # Positive electrode
        current_pos = butler_volmer_lead_acid(c_p, phi_p, domain=["positive electrode"])
        # Concatenate
        return pybamm.Concatenation(current_neg, current_sep, current_pos)
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))


def exchange_current_density_lead_acid(c_e, domain=None):
    """The exchange current-density as a function of concentration

    Parameters
    ----------
    c_e : :class:`pybamm.Variable`
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
        if c_e.domain != []:
            domain = c_e.domain
        # otherwise raise error
        else:
            raise ValueError("domain cannot be None if c.domain is empty")
    # concentration domain should be empty or the same as domain
    if c_e.domain not in [domain, []]:
        raise pybamm.DomainError("""concentration and domain do not match""")

    sp = pybamm.standard_parameters
    if domain == ["negative electrode"]:
        return sp.m_n * c_e
    elif domain == ["positive electrode"]:
        c_w = (1 - c_e * sp.V_e) / sp.V_w
        return sp.m_p * c_e ** 2 * c_w
    else:
        raise pybamm.DomainError("domain '{}' not recognised".format(domain))
