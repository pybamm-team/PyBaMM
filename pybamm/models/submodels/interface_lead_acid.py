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
        raise ValueError("Oxygen reaction uses Tafel kinetics instead of Butler-Volmer")

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
        if domain == ["positive electrode"]:
            # Only forward reaction really contributes (eta_r >> 0)
            return j0 * pybamm.exp((param.ne_Ox / 2) * eta_r)

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

        if domain == ["positive electrode"]:
            return param.j0_p_Ox_ref * c_e  # ** param.exponent_e_Ox

    def get_derived_interfacial_currents(self, j_n, j_p, j0_n, j0_p, reaction="oxygen"):
        """
        See
        :meth:`pybamm.interface.InterfacialReaction.get_derived_interfacial_currents`
        """
        return super().get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p, reaction)


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

    def get_variables(self, curlyU_n, curlyU_p):
        param = self.set_of_parameters

        # Broadcast if necessary
        if curlyU_n.domain in [[], ["current collector"]]:
            curlyU_n = pybamm.Broadcast(curlyU_n, ["negative electrode"])
        if curlyU_p.domain in [[], ["current collector"]]:
            curlyU_p = pybamm.Broadcast(curlyU_p, ["positive electrode"])

        a_n_S = self.get_interfacial_surface_area(curlyU_n, "main")
        a_p_S = self.get_interfacial_surface_area(curlyU_p, "main")
        a_n_Ox = self.get_interfacial_surface_area(curlyU_n, "oxygen")
        a_p_Ox = self.get_interfacial_surface_area(curlyU_p, "oxygen")

        soc = " utilisation"
        main_area = " surface area density (main reaction)"
        ox_area = " surface area density (oxygen reaction)"
        variables = {}
        for domain, curlyU, a_S, a_Ox, a_scale in [
            ["negative electrode", curlyU_n, a_n_S, a_n_Ox, param.a_n_dim],
            ["positive electrode", curlyU_p, a_p_S, a_p_Ox, param.a_p_dim],
        ]:
            domain_variables = {
                domain.capitalize() + soc: curlyU,
                "Average " + domain + soc: pybamm.average(curlyU),
                domain.capitalize() + main_area: a_S,
                "Average " + domain + main_area: pybamm.average(a_S),
                domain.capitalize() + ox_area: a_Ox,
                "Average " + domain + ox_area: pybamm.average(a_Ox),
                domain.capitalize() + main_area + " [m-1]": a_scale * a_S,
                "Average "
                + domain
                + main_area
                + " [m-1]": a_scale * pybamm.average(a_S),
                domain.capitalize() + ox_area + " [m-1]": a_scale * a_Ox,
                "Average "
                + domain
                + ox_area
                + " [m-1]": a_scale * pybamm.average(a_Ox),
            }
            variables.update(domain_variables)
        return variables

    def get_current_variables(self, variables):

        main_current_per_volume = " interfacial current density per volume"
        main_current = " interfacial current density"
        main_area = " surface area density (main reaction)"
        ox_current_per_volume = " oxygen interfacial current density per volume"
        ox_current = " oxygen interfacial current density"
        ox_area = " surface area density (oxygen reaction)"
        new_variables = {}
        for domain in ["negative electrode", "positive electrode"]:
            j = variables[domain.capitalize() + main_current]
            a_S = variables[domain.capitalize() + main_area]
            j_bar = variables["Average " + domain + main_current]
            a_S_bar = variables["Average " + domain + main_area]
            # Get variables for oxygen if they exist, otherwise set to zero
            j_Ox = variables.get(domain.capitalize() + ox_current, pybamm.Scalar(0))
            a_Ox = variables.get(domain.capitalize() + ox_area, pybamm.Scalar(0))
            j_Ox_bar = variables.get("Average " + domain + ox_current, pybamm.Scalar(0))
            a_Ox_bar = variables.get("Average " + domain + ox_area, pybamm.Scalar(0))

            domain_variables = {
                domain.capitalize() + main_current_per_volume: a_S * j,
                "Average " + domain + main_current_per_volume: a_S_bar * j_bar,
                domain.capitalize() + ox_current_per_volume: a_Ox * j_Ox,
                "Average " + domain + ox_current_per_volume: a_Ox_bar * j_Ox_bar,
            }
            new_variables.update(domain_variables)
        return new_variables


class VaryingSurfaceArea(InterfacialSurfaceArea):
    """
    Varying interfacial surface area

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.interface_lead_acid.InterfacialSurfaceArea`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, variables, domain, leading_order=False):
        param = self.set_of_parameters
        curlyU_n = variables["Negative electrode utilisation"]
        curlyU_p = variables["Positive electrode utilisation"]
        self.variables = self.get_variables(curlyU_n, curlyU_p)
        if domain == ["negative electrode"]:
            curlyU = curlyU_n
            j = variables["Negative electrode interfacial current density"]
            beta_U = param.beta_U_n
            curlyU_init = param.curlyU_n_init
        elif domain == ["positive electrode"]:
            curlyU = curlyU_p
            j = variables["Positive electrode interfacial current density"]
            beta_U = param.beta_U_p
            curlyU_init = param.curlyU_p_init

        if leading_order:
            j = j.orphans[0]

        # Create model
        a = self.get_interfacial_surface_area(curlyU, "main")
        self.rhs[curlyU] = beta_U * a * j
        self.initial_conditions[curlyU] = curlyU_init
        # Events: cut off if curlyU hits zero or one, with some tolerance for the
        # fact that the initial conditions can be curlyU = 0
        # self.events = [pybamm.min(curlyU) + 0.0001, pybamm.max(curlyU) - 1]

    def get_interfacial_surface_area(self, curlyU, reaction):
        param = self.set_of_parameters
        if reaction == "main":
            return curlyU ** param.xi
        elif reaction == "oxygen":
            return 1 - curlyU ** param.xi


class ConstantSurfaceArea(InterfacialSurfaceArea):
    """
    Constant interfacial surface area

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.interface_lead_acid.InterfacialSurfaceArea`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_interfacial_surface_area(self, curlyU=None, reaction=None):
        return pybamm.Scalar(1)
