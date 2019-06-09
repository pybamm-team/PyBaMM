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

    def get_butler_volmer(self, j0, eta_r, domain=None):
        raise NotImplementedError(
            "Oxygen reaction uses Tafel kinetics instead of Butler-Volmer"
        )

    def get_tafel(self, j0, eta_r, domain=None):
        """
        Tafel kinetics for the oxygen reaction

        Parameters
        ----------
        j0a : :class:`pybamm.Symbol`
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
            # Only backward reaction really contributes (eta_r << 0)
            return -j0 * pybamm.exp((param.ne_n / 2) * -eta_r)
        elif domain == ["positive electrode"]:
            # Only forward reaction really contributes (eta_r >> 0)
            return j0 * pybamm.exp((param.ne_Ox / 2) * eta_r)
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_mass_transfer_limited(self, c_ox):
        return -c_ox

    def get_exchange_current_densities(self, c_e, c_ox, domain=None):
        """The exchange current-density as a function of concentrations

        Parameters
        ----------
        c_e : :class:`pybamm.Symbol`
            Electrolyte concentration
        c_ox : :class:`pybamm.Symbol`
            Oxygen concentration
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
            return param.j0_p_Ox_ref * c_e  # ** param.exponent_e_Ox

    def get_derived_interfacial_currents(self, j_n, j_p, j0_n, j0_p):
        """
        See
        :meth:`pybamm.interface.InterfacialReaction.get_derived_interfacial_currents`
        """
        return super().get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p, "oxygen")


class InterfacialSurfaceArea(pybamm.SubModel):
    """
    Base class for interfacial surface area

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, variables, domain=None):
        param = self.set_of_parameters
        domain = domain or j.domain
        if domain == ["negative electrode"]:
            curlyU = self.variables["Negative electrode State of Charge"]
            j = self.variables["Negative electrode interfacial current density"]
            beta_U = param.beta_U_n
            curlyU_init = param.curlyU_n_init
        elif domain == ["negative electrode"]:
            curlyU = self.variables["Positive electrode State of Charge"]
            j = self.variables["Positive electrode interfacial current density"]
            beta_U = param.beta_U_p
            curlyU_init = param.curlyU_p_init

        # Create model
        self.rhs = {curlyU: beta_U * j}
        self.initial_conditions = {curlyU: curlyU_init}

        # Events: cut off if curlyU hits zero or one, with some tolerance for the
        # fact that the initial conditions can be curlyU = 0
        self.events = [pybamm.min(curlyU) + 0.0001, pybamm.max(curlyU) - 1]
