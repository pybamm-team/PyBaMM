#
# Equations for the electrode-electrolyte interface for lead-acid models
#
import pybamm


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
            return param.m_n * c_e
        elif domain == ["positive electrode"]:
            c_w = param.c_w(c_e)
            return param.m_p * c_e ** 2 * c_w
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

    def get_exchange_current_densities(self, c_e, domain=None):
        """The exchange current-density as a function of concentrations

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
