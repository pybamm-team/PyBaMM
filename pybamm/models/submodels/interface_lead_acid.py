#
# Equations for the electrode-electrolyte interface for lead-acid models
#
import pybamm
import autograd.numpy as np


class MainReaction(pybamm.interface.InterfacialReaction, pybamm.LeadAcidBaseModel):
    """
    Main lead-acid reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`InterfacialReaction`, :class:`pybamm.LeadAcidBaseModel`
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
            return param.j0_n_S_ref * c_e
        elif domain == ["positive electrode"]:
            c_w = param.c_w(c_e)
            return param.j0_p_S_ref * c_e ** 2 * c_w
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))


class OxygenReaction(pybamm.interface.InterfacialReaction, pybamm.LeadAcidBaseModel):
    """
    Oxygen reaction in lead-acid batteries

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`InterfacialReaction`, :class:`pybamm.LeadAcidBaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_butler_volmer(self, j0a, j0c, eta_r, domain=None):
        """
        Butler-Volmer kinetics for the oxygen reaction

        Parameters
        ----------
        j0a : :class:`pybamm.Symbol`
            Exchange-current density (forward reaction)
        j0c : : class:`pybamm.Symbol`
            Exchange-current density (backward reaction)
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

        domain = domain or j0a.domain
        if domain == ["negative electrode"]:
            forward = j0a * pybamm.exp((param.ne_n / 2) * eta_r)
            backward = j0c * pybamm.exp((param.ne_n / 2) * -eta_r)
            return forward - backward
        elif domain == ["positive electrode"]:
            forward = j0a * pybamm.exp((param.ne_p / 2) * eta_r)
            backward = j0c * pybamm.exp((param.ne_p / 2) * -eta_r)
            return forward - backward
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_exchange_current_densities(self, c_e, c_ox, direction, domain=None):
        """The exchange current-density as a function of concentrations

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        c_ox : :class:`pybamm.Symbol`
            Oxygen concentration
        direction : str
            Whether to take the forward or backward reaction
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
            return param.j0_n_Ox_ref * c_e
        elif domain == ["positive electrode"]:
            common = param.j0_p_Ox_ref * c_e  # ** param.exponent_e_Ox
            if direction == "forward":
                return common
            elif direction == "backward":
                return common  # * c_ox ** param.exponent_ox_Ox
