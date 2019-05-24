#
# Equations for the electrode-electrolyte interface
#
import pybamm
import autograd.numpy as np


class InterfacialCurrent(pybamm.SubModel):
    """
    Base class for interfacial currents

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_homogeneous_interfacial_current(self, domain):
        """
        Homogeneous reaction at the electrode-electrolyte interface

        Parameters
        ----------
        domain : iter of str
            The domain(s) in which to compute the interfacial current.

        Returns
        -------
        :class:`pybamm.Symbol`
            Homogeneous interfacial current density
        """
        icell = pybamm.electrical_parameters.current_with_time

        if domain == ["negative electrode"]:
            return icell / pybamm.geometric_parameters.l_n
        elif domain == ["positive electrode"]:
            return -icell / pybamm.geometric_parameters.l_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_butler_volmer(self, j0, eta_r, domain=None):
        """
        Butler-Volmer reactions

        .. math::
            j = j_0(c) * \\sinh(\\eta_r(c))

        Parameters
        ----------
        j0 : :class:`pybamm.Symbol`
            Exchange-current density
        eta_r : :class:`pybamm.Symbol`
            Reaction overpotential
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case j0.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Interfacial current density

        """
        param = self.set_of_parameters

        domain = domain or j0.domain
        if domain == ["negative electrode"]:
            return 2 * j0 * pybamm.Function(np.sinh, (param.ne_n / 2) * eta_r)
        elif domain == ["positive electrode"]:
            return 2 * j0 * pybamm.Function(np.sinh, (param.ne_p / 2) * eta_r)
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_inverse_butler_volmer(self, j, j0, domain=None):
        """
        Inverts the Butler-Volmer relation to solve for the reaction overpotential.

        Parameters
        ----------
        j : :class:`pybamm.Symbol`
            Interfacial current density
        j0 : :class:`pybamm.Symbol`
            Exchange-current density
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case j.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Reaction overpotential

        """
        param = self.set_of_parameters

        domain = domain or j.domain
        if domain == ["negative electrode"]:
            return (2 / param.ne_n) * pybamm.Function(np.arcsinh, j / (2 * j0))
        elif domain == ["positive electrode"]:
            return (2 / param.ne_p) * pybamm.Function(np.arcsinh, j / (2 * j0))
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_derived_interfacial_currents(self, j_n, j_p, j0_n, j0_p):
        """
        Calculate dimensionless and dimensional variables for the interfacial current
        submodel

        Parameters
        ----------
        j_n : :class:`pybamm.Symbol`
            Interfacial current density in the negative electrode
        j_p : :class:`pybamm.Symbol`
            Interfacial current density in the positive electrode
        j0_n : :class:`pybamm.Symbol`
            Exchange-current density in the negative electrode
        j0_p : :class:`pybamm.Symbol`
            Exchange-current density in the positive electrode

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        i_typ = self.set_of_parameters.i_typ

        # Broadcast if necessary
        if j_n.domain in [[], ["current collector"]]:
            j_n = pybamm.Broadcast(j_n, ["negative electrode"])
        if j_p.domain in [[], ["current collector"]]:
            j_p = pybamm.Broadcast(j_p, ["positive electrode"])
        if j0_n.domain in [[], ["current collector"]]:
            j0_n = pybamm.Broadcast(j0_n, ["negative electrode"])
        if j0_p.domain in [[], ["current collector"]]:
            j0_p = pybamm.Broadcast(j0_p, ["positive electrode"])

        # Concatenations
        j = pybamm.Concatenation(*[j_n, pybamm.Broadcast(0, ["separator"]), j_p])
        j0 = pybamm.Concatenation(*[j0_n, pybamm.Broadcast(0, ["separator"]), j0_p])

        # Averages
        j_n_av = pybamm.average(j_n)
        j_p_av = pybamm.average(j_p)

        return {
            "Negative electrode interfacial current density": j_n,
            "Positive electrode interfacial current density": j_p,
            "Average negative electrode interfacial current density": j_n_av,
            "Average positive electrode interfacial current density": j_p_av,
            "Interfacial current density": j,
            "Negative electrode exchange-current density": j0_n,
            "Positive electrode exchange-current density": j0_p,
            "Exchange-current density": j0,
            "Negative electrode interfacial current density [A.m-2]": i_typ * j_n,
            "Positive electrode interfacial current density [A.m-2]": i_typ * j_p,
            "Average negative electrode interfacial current density [A.m-2]": i_typ
            * j_n_av,
            "Average positive electrode interfacial current density [A.m-2]": i_typ
            * j_p_av,
            "Interfacial current density [A.m-2]": i_typ * j,
            "Negative electrode exchange-current density [A.m-2]": i_typ * j0_n,
            "Positive electrode exchange-current density [A.m-2]": i_typ * j0_p,
            "Exchange-current density [A.m-2]": i_typ * j0,
        }


class LeadAcidReaction(InterfacialCurrent, pybamm.LeadAcidBaseModel):
    """
    Interfacial current from lead-acid reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`InterfacialCurrent`, :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_exchange_current_densities(self, c_e, domain=None):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case c_e.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density

        """
        param = self.set_of_parameters
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            return param.m_n * c_e
        elif domain == ["positive electrode"]:
            c_w = param.c_w(c_e)
            return param.m_p * c_e ** 2 * c_w
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))


class LithiumIonReaction(InterfacialCurrent):
    """
    Interfacial current from lithium-ion reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`InterfacialCurrent`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_exchange_current_densities(self, c_e, c_s_k_surf, domain=None):
        """The exchange current-density as a function of concentration

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        c_s_k_surf : :class:`pybamm.Symbol`
            Electrode surface concentration
        domain : iter of str, optional
            The domain(s) in which to compute the interfacial current. Default is None,
            in which case c_e.domain is used.

        Returns
        -------
        :class:`pybamm.Symbol`
            Exchange-current density
        """
        param = self.set_of_parameters
        domain = domain or c_e.domain

        if domain == ["negative electrode"]:
            return (1 / param.C_r_n) * (
                c_e ** (1 / 2) * c_s_k_surf ** (1 / 2) * (1 - c_s_k_surf) ** (1 / 2)
            )
        elif domain == ["positive electrode"]:
            return (param.gamma_p / param.C_r_p) * (
                c_e ** (1 / 2) * c_s_k_surf ** (1 / 2) * (1 - c_s_k_surf) ** (1 / 2)
            )
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))
