#
# Equation classes for the electrolyte current
#
import pybamm


class ElectrolyteCurrentBaseModel(pybamm.SubModel):
    """
    Base model for the potential and current in the electrolyte

    **Extends:** :class:`pybamm.SubModel`

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def unpack(self, variables, domain=None):
        """
        Unpack a dictionary of variables.

        Parameters
        ----------
        variables : dict
            Dictionary of variables to unpack
        domain : list of str
            Domain in which to unpack the variables
        """
        i_boundary_cc = variables["Current collector current density"]
        delta_phi_n = variables.get("Negative electrode surface potential difference")
        delta_phi_p = variables.get("Positive electrode surface potential difference")
        c_e = variables.get("Electrolyte concentration")
        if c_e is None or isinstance(c_e, pybamm.Variable):
            c_e_n, c_e_s, c_e_p = c_e, c_e, c_e
        else:
            c_e_n, c_e_s, c_e_p = c_e.orphans
        eps = variables.get("Porosity")
        eps = eps or self.set_of_parameters.epsilon
        eps_n, eps_s, eps_p = eps.orphans

        if domain == ["negative electrode"]:
            return i_boundary_cc, delta_phi_n, c_e_n, eps_n
        elif domain == ["positive electrode"]:
            return i_boundary_cc, delta_phi_p, c_e_p, eps_p
        else:
            return i_boundary_cc, (delta_phi_n, delta_phi_p), c_e, eps

    def get_explicit_leading_order(self, variables):
        """
        Provides explicit leading order solution to the electrolyte current conservation
        equation where the constitutive equation is taken to be of Stefan-Maxwell form.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # unpack variables
        ocp_n = variables["Negative electrode open circuit potential"]
        eta_r_n = variables["Negative electrode reaction overpotential"]
        i_boundary_cc = variables["Current collector current density"]

        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # electrolye potential
        phi_e_const = -pybamm.boundary_value(ocp_n, "left") - pybamm.boundary_value(
            eta_r_n, "left"
        )
        phi_e_n = pybamm.Broadcast(phi_e_const, ["negative electrode"])
        phi_e_s = pybamm.Broadcast(phi_e_const, ["separator"])
        phi_e_p = pybamm.Broadcast(phi_e_const, ["positive electrode"])
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # electrolyte current
        i_e_n = pybamm.outer(i_boundary_cc, x_n / l_n)
        i_e_p = pybamm.outer(i_boundary_cc, (1 - x_p) / l_p)

        # electrolyte ohmic losses
        delta_phi_e_av = pybamm.Scalar(0)
        # concentration overpotential
        eta_c_av = pybamm.Scalar(0)
        # electrolyte overpotential
        eta_e_av = eta_c_av + delta_phi_e_av

        pot_variables = self.get_potential_variables(phi_e, eta_e_av)
        current_variables = self.get_current_variables((i_e_n, i_e_p), i_boundary_cc)
        additional_vars = self.get_split_electrolyte_overpotential(
            eta_c_av, delta_phi_e_av
        )

        return {**pot_variables, **current_variables, **additional_vars}

    def get_explicit_combined(self, variables, first_order="composite"):
        """
        Provides an explicit combined leading and first order solution to the
        electrolyte current conservation equation where the constitutive equation is
        taken to be of Stefan-Maxwell form. Note that the returned current density is
        only the leading order approximation.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        first_order : str
            Whether to take the linear correction to first-order, or the composite one.
            Default is "composite".

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """

        def first_order_function(c_e):
            if first_order == "composite":
                return pybamm.log(c_e)
            elif first_order == "linear":
                return c_e

        # unpack variables
        i_boundary_cc, _, c_e, epsilon = self.unpack(variables)
        ocp_n = variables["Negative electrode open circuit potential"]
        eta_r_n = variables["Negative electrode reaction overpotential"]
        phi_s_n = variables["Negative electrode potential"]
        # Get average electrolyte concentration if it exists, otherwise set to 1
        c_e_0 = variables.get("Average electrolyte concentration", pybamm.Scalar(1))

        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # extract c_e components
        c_e_n, c_e_s, c_e_p = c_e.orphans

        # if porosity is not provided, use the input parameter
        if c_e_0 is None:
            c_e_0 = pybamm.Scalar(1)
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # bulk conductivities (leading order)
        kappa_n = param.kappa_e(c_e_0) * eps_n ** param.b
        kappa_s = param.kappa_e(c_e_0) * eps_s ** param.b
        kappa_p = param.kappa_e(c_e_0) * eps_p ** param.b
        chi_0 = param.chi(c_e_0)

        # get electrode averaged values
        ocp_n_av = pybamm.average(ocp_n)
        eta_r_n_av = pybamm.average(eta_r_n)
        phi_s_n_av = pybamm.average(phi_s_n)

        # electrolyte current (leading-order approximation)
        i_e_n = pybamm.outer(i_boundary_cc, x_n / l_n)
        i_e_p = pybamm.outer(i_boundary_cc, (1 - x_p) / l_p)

        # electrolyte potential (combined leading and first order)
        phi_e_const = (
            (-ocp_n_av - eta_r_n_av + phi_s_n_av)
            - chi_0 * pybamm.average(first_order_function(c_e_n / c_e_0))
            - (
                (i_boundary_cc * param.C_e * l_n / param.gamma_e)
                * (1 / (3 * kappa_n) - 1 / kappa_s)
            )
        )

        phi_e_n = (
            phi_e_const
            + chi_0 * first_order_function(c_e_n / c_e_0)
            - (i_boundary_cc * param.C_e / param.gamma_e)
            * ((x_n ** 2 - l_n ** 2) / (2 * kappa_n * l_n) + l_n / kappa_s)
        )

        phi_e_s = (
            phi_e_const
            + chi_0 * first_order_function(c_e_s / c_e_0)
            - (i_boundary_cc * param.C_e / param.gamma_e) * (x_s / kappa_s)
        )

        phi_e_p = (
            phi_e_const
            + chi_0 * first_order_function(c_e_p / c_e_0)
            - (i_boundary_cc * param.C_e / param.gamma_e)
            * (
                (x_p * (2 - x_p) + l_p ** 2 - 1) / (2 * kappa_p * l_p)
                + (1 - l_p) / kappa_s
            )
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        "Ohmic losses and overpotentials"
        # average electrolyte ohmic losses
        delta_phi_e_av = -(param.C_e * i_boundary_cc / param.gamma_e) * (
            param.l_n / (3 * kappa_n)
            + param.l_s / (kappa_s)
            + param.l_p / (3 * kappa_p)
        )

        # concentration overpotential (combined leading and first order)
        eta_c_av = chi_0 * (
            pybamm.average(first_order_function(c_e_p / c_e_0))
            - pybamm.average(first_order_function(c_e_n / c_e_0))
        )

        # electrolyte overpotential
        eta_e_av = eta_c_av + delta_phi_e_av

        # get variables
        pot_variables = self.get_potential_variables(phi_e, eta_e_av)
        current_variables = self.get_current_variables((i_e_n, i_e_p), i_boundary_cc)
        additional_vars = self.get_split_electrolyte_overpotential(
            eta_c_av, delta_phi_e_av
        )

        return {**pot_variables, **current_variables, **additional_vars}

    def get_potential_variables(self, phi_e, eta_e_av):
        """
        Calculate dimensionless and dimensional variables for the electrolyte current
        submodel

        Parameters
        ----------
        phi_e :class:`pybamm.Concatenation`
            The electrolyte potential
        delta_phi_e_av: :class:`pybamm.Symbol`
            Average Ohmic losses in the electrolyte
        eta_e_av: :class:`Pybamm.Symbol`
            Average electrolyte overpotential

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        param = self.set_of_parameters
        pot_scale = param.potential_scale

        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        phi_e_av = pybamm.average(phi_e)

        # Set dimensionless and dimensional variables
        return {
            "Negative electrolyte potential": phi_e_n,
            "Separator electrolyte potential": phi_e_s,
            "Positive electrolyte potential": phi_e_p,
            "Electrolyte potential": phi_e,
            "Average electrolyte overpotential": eta_e_av,
            "Negative electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_n,
            "Separator electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_s,
            "Positive electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_p,
            "Electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e,
            "Average electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_av,
            "Average electrolyte overpotential [V]": pot_scale * eta_e_av,
        }

    def get_current_variables(self, i_e, i_boundary_cc=None):
        """
        Calculate dimensionless and dimensional current variables.

        Parameters
        ----------
        i_e :class:`pybamm.Symbol`
            The electrolyte current density

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        i_typ = self.set_of_parameters.i_typ
        if isinstance(i_e, tuple):
            i_e_n, i_e_p = i_e
            i_e_s = pybamm.Broadcast(i_boundary_cc, "separator")
            i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        return {
            "Electrolyte current density": i_e,
            "Electrolyte current density [A.m-2]": i_typ * i_e,
        }

    def get_split_electrolyte_overpotential(self, eta_c_av, delta_phi_e_av):
        param = self.set_of_parameters
        pot_scale = param.potential_scale

        return {
            "Average concentration overpotential": eta_c_av,
            "Average electrolyte ohmic losses": delta_phi_e_av,
            "Average concentration overpotential [V]": pot_scale * eta_c_av,
            "Average electrolyte ohmic losses [V]": pot_scale * delta_phi_e_av,
        }


class MacInnesStefanMaxwell(ElectrolyteCurrentBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations.

    **Extends:** :class:`ElectrolyteCurrentBaseModel`

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_algebraic_system(self, variables, reactions):
        """
        PDE system for current in the electrolyte, derived from the Stefan-Maxwell
        equations.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        """
        # Load parameters
        param = self.set_of_parameters

        # Unpack variables
        phi_e = variables["Electrolyte potential"]
        c_e = variables["Electrolyte concentration"]
        try:
            epsilon = variables["Porosity"]
        except KeyError:
            epsilon = param.epsilon

        # Unpack variables
        j_n = reactions["main"]["neg"]["aj"]
        j_p = reactions["main"]["pos"]["aj"]
        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)

        # functions
        i_e = (
            param.kappa_e(c_e) * (epsilon ** param.b) * param.gamma_e / param.C_e
        ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e))

        # Equations (algebraic only)
        self.algebraic = {phi_e: pybamm.div(i_e) - j}
        self.boundary_conditions = {
            phi_e: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        self.initial_conditions = {phi_e: -param.U_n(param.c_n_init)}
        # no differential equations
        self.rhs = {}

        # Variables
        # average electrolyte overpotential (ohmic + concentration overpotential)
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        phi_e_n_av = pybamm.average(phi_e_n)
        phi_e_p_av = pybamm.average(phi_e_p)
        eta_e_av = phi_e_p_av - phi_e_n_av

        self.variables.update(self.get_potential_variables(phi_e, eta_e_av))
        self.variables.update(self.get_current_variables(i_e))

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()


class MacInnesCapacitance(ElectrolyteCurrentBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations, with capacitance effects included. The MacInnes equation
    is rearranged to account for both solid and electrolyte potentials

    **Extends:** :class:`ElectrolyteCurrentBaseModel`

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel
    capacitance_options : str
        The type of equation to set for capacitance. Can be "differential" (default) or
        "algebraic"
    """

    def __init__(self, set_of_parameters, capacitance_options="differential"):
        super().__init__(set_of_parameters)
        self._capacitance_options = capacitance_options

    @property
    def capacitance_options(self):
        return self._capacitance_options

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self._capacitance_options == "differential":
            default_solver = pybamm.ScipySolver()
        else:
            default_solver = pybamm.ScikitsDaeSolver()

        return default_solver

    def set_full_system(self, variables, reactions, domain):
        """
        PDE system for current in the electrolyte, derived from the Stefan-Maxwell
        equations. If capacitance_options is `differential`, this adds equations to
        `rhs`. Otherwise, this adds equations to `algebraic`

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        domain : list of str
            Domain in which to set the system
        """
        param = self.set_of_parameters
        _, delta_phi, c_e, eps = self.unpack(variables, domain)

        if domain == ["negative electrode"]:
            j = reactions["main"]["neg"]["aj"]
            self.initial_conditions[delta_phi] = param.U_n(param.c_n_init)
            C_dl = param.C_dl_n
            Domain = "Negative"
        elif domain == ["positive electrode"]:
            j = reactions["main"]["pos"]["aj"]
            self.initial_conditions[delta_phi] = param.U_p(param.c_p_init)
            C_dl = param.C_dl_p
            Domain = "Positive"
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))
        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e
        i_e = conductivity * (
            (param.chi(c_e) / c_e) * pybamm.grad(c_e) + pybamm.grad(delta_phi)
        )
        if self.capacitance_options == "differential":
            self.rhs[delta_phi] = 1 / C_dl * (pybamm.div(i_e) - j)
        elif self.capacitance_options == "algebraic":
            self.algebraic[delta_phi] = pybamm.div(i_e) - j

        # Set boundary conditons and variables
        self.set_boundary_conditions(variables, conductivity, domain)
        self.variables.update(variables)
        self.variables[Domain + " electrolyte current density"] = i_e

    def set_boundary_conditions(self, variables, conductivity, domain):
        """
        Set boundary conditions for the full model.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        conductivity : :class:`pybamm.Symbol`
            Effective electrolyte conductivity
        domain : list of str
            Domain in which to set the system
        """
        param = self.set_of_parameters
        i_boundary_cc, delta_phi, c_e, _ = self.unpack(variables, domain)

        flux_bc_side = {"negative": "right", "positive": "left"}[domain[0][:8]]
        other_side = {"left": "right", "right": "left"}[flux_bc_side]

        c_e_flux = pybamm.BoundaryFlux(c_e, flux_bc_side)
        flux_bc = (
            i_boundary_cc / pybamm.BoundaryValue(conductivity, flux_bc_side)
        ) - pybamm.BoundaryValue(param.chi(c_e) / c_e, flux_bc_side) * c_e_flux
        self.boundary_conditions[delta_phi] = {
            flux_bc_side: (flux_bc, "Neumann"),
            other_side: (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[c_e] = {
            flux_bc_side: (c_e_flux, "Neumann"),
            other_side: (pybamm.Scalar(0), "Neumann"),
        }

    def set_leading_order_system(self, variables, reactions, domain):
        """
        ODE system for leading-order current in the electrolyte, derived from the
        Stefan-Maxwell equations. If capacitance_options is `differential`, this adds
        equations to `rhs`. Otherwise, this adds equations to `algebraic`.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        reactions : dict
            Dictionary of reaction variables
        domain : list of str
            Domain in which to set the system
        """
        param = self.set_of_parameters
        (i_boundary_cc, delta_phi_broad, *rest) = self.unpack(variables, domain)
        delta_phi = delta_phi_broad.child

        if domain == ["negative electrode"]:
            j_average = i_boundary_cc / param.l_n
            j = sum(reaction["neg"]["aj"] for reaction in reactions.values())
            self.initial_conditions[delta_phi] = param.U_n(param.c_n_init)
            C_dl = param.C_dl_n
        elif domain == ["positive electrode"]:
            j_average = -i_boundary_cc / param.l_p
            j = sum(reaction["pos"]["aj"] for reaction in reactions.values())
            self.initial_conditions[delta_phi] = param.U_p(param.c_p_init)
            C_dl = param.C_dl_p
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

        if self.capacitance_options == "differential":
            self.rhs[delta_phi] = 1 / C_dl * (j_average - j)
        elif self.capacitance_options == "algebraic":
            self.algebraic[delta_phi] = j_average - j

    def set_post_processed(self):
        """
        Calculate dimensionless and dimensional variables for the capacitance submodel

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack variables
        i_boundary_cc, (delta_phi_n, delta_phi_p), c_e, eps = self.unpack(
            self.variables
        )
        c_e_n, c_e_s, c_e_p = c_e.orphans
        eps_n, eps_s, eps_p = eps.orphans

        # Unpack and combine currents
        i_e_n = self.variables["Negative electrolyte current density"]
        i_e_s = pybamm.Broadcast(i_boundary_cc, ["separator"])
        i_e_p = self.variables["Positive electrolyte current density"]
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)
        i_s_n = i_boundary_cc - i_e_n
        i_s_p = i_boundary_cc - i_e_p

        # Negative electrode potential
        solid_conductivity_n = param.sigma_n * (1 - eps_n) ** param.b
        phi_s_n = -pybamm.IndefiniteIntegral(i_s_n / solid_conductivity_n, x_n)
        # Separator electrolyte potential
        phi_e_n = phi_s_n - delta_phi_n
        chi_s = param.chi(c_e_s)
        kappa_s_eff = param.kappa_e(c_e_s) * (eps_s ** param.b)
        phi_e_s = pybamm.boundary_value(phi_e_n, "right") + pybamm.IndefiniteIntegral(
            chi_s / c_e_s * pybamm.grad(c_e_s)
            - param.C_e * i_boundary_cc / kappa_s_eff,
            x_s,
        )
        # Positive electrode potential
        solid_conductivity_p = param.sigma_p * (1 - eps_p) ** param.b
        phi_s_p = (
            -pybamm.IndefiniteIntegral(i_s_p / solid_conductivity_p, x_p)
            + pybamm.boundary_value(phi_e_s, "right")
            + pybamm.boundary_value(delta_phi_p, "left")
        )
        phi_e_p = phi_s_p - delta_phi_p

        # Concatenate
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # average elecrolyte overpotential (ohmic + concentration overpotential)
        phi_e_n_av = pybamm.average(phi_e_n)
        phi_e_p_av = pybamm.average(phi_e_p)
        eta_e_av = phi_e_p_av - phi_e_n_av

        # Update variables
        self.variables.update(self.get_potential_variables(phi_e, eta_e_av))
        self.variables.update(self.get_current_variables(i_e))
        electrode_current_model = pybamm.electrode.Ohm(param)
        pot_vars = electrode_current_model.get_potential_variables(phi_s_n, phi_s_p)
        curr_vars = electrode_current_model.get_current_variables(i_s_n, i_s_p)
        self.variables.update({**pot_vars, **curr_vars})

        # Update boundary conditions (for indefinite integral)
        self.boundary_conditions[c_e_s] = {
            "left": (pybamm.BoundaryFlux(c_e_s, "left"), "Neumann"),
            "right": (pybamm.BoundaryFlux(c_e_s, "right"), "Neumann"),
        }
