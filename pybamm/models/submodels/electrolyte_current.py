#
# Equation classes for the electrolyte current
#
import pybamm

import numpy as np


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

    def get_explicit_leading_order(self, ocp_n, eta_r_n):
        """
        Provides explicit leading order solution to the electrolyte current conservation
        equation where the constitutive equation is taken to be of Stefan-Maxwell form.

        Parameters
        ----------
        ocp_n : :class:`pybamm.Symbol`
            Open-circuit potential in the negative electrode
        eta_r_n : :class:`pybamm.Symbol`
            Reaction overpotential in the negative electrode

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # define current
        i_cell = param.current_with_time

        # electrolye potential
        phi_e_const = -ocp_n - eta_r_n
        phi_e_n = pybamm.Broadcast(phi_e_const, ["negative electrode"])
        phi_e_s = pybamm.Broadcast(phi_e_const, ["separator"])
        phi_e_p = pybamm.Broadcast(phi_e_const, ["positive electrode"])
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # electrolyte current
        i_e_n = i_cell * x_n / l_n
        i_e_s = pybamm.Broadcast(i_cell, ["separator"])
        i_e_p = i_cell * (1 - x_p) / l_p
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        # electrolyte ohmic losses
        delta_phi_e_av = pybamm.Scalar(0)
        # concentration overpotential
        eta_c_av = pybamm.Scalar(0)
        # electrolyte overpotential
        eta_e_av = eta_c_av + delta_phi_e_av

        variables = self.get_variables(phi_e, i_e, eta_e_av)
        additional_vars = self.get_split_electrolyte_overpotential(
            eta_c_av, delta_phi_e_av
        )
        variables.update(additional_vars)

        return variables

    def get_explicit_combined(
        self, ocp_n, eta_r_n, c_e, phi_s_n, epsilon=None, c_e_0=None
    ):
        """
        Provides and explicit combined leading and first order solution to the
        electrolyte current conservation equation where the constitutive equation is
        taken to be of Stefan-Maxwell form. Note that the returned current density is
        only the leading order approximation.

        Parameters
        ----------
        ocp_n : :class:`pybamm.Symbol`
            Open-circuit potential in the negative electrode
        eta_r_n : :class:`pybamm.Symbol`
            Reaction overpotential in the negative electrode
        c_e : :class:`pybamm.Concatenation`
            The electrolyte concentration variable
        phi_s_n : :class:`pybamm.Symbol`
            The negative electrode potential
        epsilon : :class:`pybamm.Symbol`, optional
            Porosity. Default is None, in which case param.epsilon is used.
        c_e : :class:`pybamm.Concatenation`
            Leading-order concentration

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # extract c_e components
        c_e_n, c_e_s, c_e_p = c_e.orphans

        # if porosity is not provided, use the input parameter
        if epsilon is None:
            epsilon = param.epsilon
        if c_e_0 is None:
            c_e_0 = pybamm.Scalar(1)
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # bulk conductivities (leading order)
        kappa_n = param.kappa_e(c_e_0) * eps_n ** param.b
        kappa_s = param.kappa_e(c_e_0) * eps_s ** param.b
        kappa_p = param.kappa_e(c_e_0) * eps_p ** param.b

        # get electrode averaged values
        ocp_n_av = pybamm.average(ocp_n)
        eta_r_n_av = pybamm.average(eta_r_n)
        phi_s_n_av = pybamm.average(phi_s_n)

        # electrolyte current (leading-order approximation)
        i_e_n = i_cell * x_n / l_n
        i_e_s = pybamm.Broadcast(i_cell, ["separator"])
        i_e_p = i_cell * (1 - x_p) / l_p
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        # electrolyte potential (combined leading and first order)
        phi_e_const = (
            -ocp_n_av
            - eta_r_n_av
            + phi_s_n_av
            - 2
            * (1 - param.t_plus)
            * pybamm.average(pybamm.Function(np.log, c_e_n / c_e_0))
            - i_cell
            * param.C_e
            * l_n
            / param.gamma_e
            * (1 / (3 * kappa_n) - 1 / kappa_s)
        )

        phi_e_n = (
            phi_e_const
            + 2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_n / c_e_0)
            - (i_cell * param.C_e / param.gamma_e)
            * ((x_n ** 2 - l_n ** 2) / (2 * kappa_n * l_n) + l_n / kappa_s)
        )

        phi_e_s = (
            phi_e_const
            + 2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_s / c_e_0)
            - (i_cell * param.C_e / param.gamma_e) * (x_s / kappa_s)
        )

        phi_e_p = (
            phi_e_const
            + 2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_p / c_e_0)
            - (i_cell * param.C_e / param.gamma_e)
            * (
                (x_p * (2 - x_p) + l_p ** 2 - 1) / (2 * kappa_p * l_p)
                + (1 - l_p) / kappa_s
            )
        )

        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        "Ohmic losses and overpotentials"
        # average electrolyte ohmic losses
        delta_phi_e_av = -(param.C_e * i_cell / param.gamma_e) * (
            param.l_n / (3 * kappa_n)
            + param.l_s / (kappa_s)
            + param.l_p / (3 * kappa_p)
        )

        # concentration overpotential (combined leading and first order)
        eta_c_av = (
            2
            * (1 - param.t_plus)
            * (
                pybamm.average(pybamm.Function(np.log, c_e_p / c_e_0))
                - pybamm.average(pybamm.Function(np.log, c_e_n / c_e_0))
            )
        )

        # electrolyte overpotential
        eta_e_av = eta_c_av + delta_phi_e_av

        # get variables
        variables = self.get_variables(phi_e, i_e, eta_e_av)
        additional_vars = self.get_split_electrolyte_overpotential(
            eta_c_av, delta_phi_e_av
        )

        variables.update(additional_vars)

        return variables

    def get_variables(self, phi_e, i_e, eta_e_av):
        """
        Calculate dimensionless and dimensional variables for the electrolyte current
        submodel

        Parameters
        ----------
        phi_e :class:`pybamm.Concatenation`
            The electrolyte potential
        i_e :class:`pybamm.Concatenation`
            The electrolyte current density
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

        # Set dimensionless and dimensional variables
        return {
            "Negative electrolyte potential": phi_e_n,
            "Separator electrolyte potential": phi_e_s,
            "Positive electrolyte potential": phi_e_p,
            "Electrolyte potential": phi_e,
            "Electrolyte current density": i_e,
            "Average electrolyte overpotential": eta_e_av,
            "Negative electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_n,
            "Separator electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_s,
            "Positive electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e_p,
            "Electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e,
            "Electrolyte current density [A.m-2]": param.i_typ * i_e,
            "Average electrolyte overpotential [V]": pot_scale * eta_e_av,
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

    def set_algebraic_system(self, phi_e, c_e, reactions, epsilon=None):
        """
        PDE system for current in the electrolyte, derived from the Stefan-Maxwell
        equations.

        Parameters
        ----------
        phi_e : :class:`pybamm.Concatenation`
            The eletrolyte potential variable
        c_e : :class:`pybamm.Concatenation`
            The eletrolyte concentration variable
        reactions : dict
            Dictionary of reaction variables
        epsilon : :class:`pybamm.Symbol`, optional
            Porosity. Default is None, in which case param.epsilon is used.
        """
        # Load parameters and spatial variables
        param = self.set_of_parameters

        # Unpack variables
        j_n = reactions["main"]["neg"]["aj"]
        j_p = reactions["main"]["pos"]["aj"]
        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)

        # if porosity is not provided, use the input parameter
        if epsilon is None:
            epsilon = param.epsilon

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

        self.variables = self.get_variables(phi_e, i_e, eta_e_av)

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
    use_capacitance : bool
        Whether to use capacitance in the model or not. If True (default), solve
        ODEs for delta_phi. If False, solve algebraic equations for delta_phi
    """

    def __init__(self, set_of_parameters, use_capacitance=True):
        super().__init__(set_of_parameters)
        self._use_capacitance = use_capacitance

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self._use_capacitance:
            default_solver = pybamm.ScikitsOdeSolver()
        else:
            default_solver = pybamm.ScikitsDaeSolver()

        return default_solver

    def set_full_system(self, delta_phi, c_e, reactions, eps=None):
        """
        PDE system for current in the electrolyte, derived from the Stefan-Maxwell
        equations. If self.use_capacitance is True, this adds equations to `rhs`.
        Otherwise, this adds equations to `algebraic`

        Parameters
        ----------
        delta_phi : :class:`pybamm.Variable`
            The potential difference variable
        c_e : :class:`pybamm.Concatenation`
            The eletrolyte concentration variable
        reactions : dict
            Dictionary of reaction variables
        epsilon : :class:`pybamm.Symbol`, optional
            Porosity. Default is None, in which case param.epsilon is used.
        """
        param = self.set_of_parameters

        if delta_phi.domain == ["negative electrode"]:
            if eps is None:
                eps = param.epsilon_n
            j = reactions["main"]["neg"]["aj"]
            C_dl = param.C_dl_n
            self.initial_conditions.update({delta_phi: param.U_n(param.c_n_init)})
            flux_bc_side = "right"
            Domain = "Negative"
        elif delta_phi.domain == ["positive electrode"]:
            if eps is None:
                eps = param.epsilon_p
            j = reactions["main"]["pos"]["aj"]
            C_dl = param.C_dl_p
            self.initial_conditions.update({delta_phi: param.U_p(param.c_n_init)})
            flux_bc_side = "left"
            Domain = "Positive"
        else:
            raise pybamm.DomainError(
                "domain '{}' not recognised".format(delta_phi.domain)
            )
        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e
        i_e = conductivity * (
            (param.chi(c_e) / c_e) * pybamm.grad(c_e) + pybamm.grad(delta_phi)
        )
        if self.use_capacitance:
            self.rhs.update({delta_phi: 1 / C_dl * (pybamm.div(i_e) - j)})
        else:
            self.algebraic.update({delta_phi: pybamm.div(i_e) - j})

        # Set boundary conditons and variables
        self.set_boundary_conditions(c_e, delta_phi, conductivity, flux_bc_side)
        self.variables.update(self.get_variables_capacitance(delta_phi, i_e, Domain))

    def set_boundary_conditions(self, c_e, delta_phi, conductivity, side):
        """ Set boundary conditions for the system. """
        param = self.set_of_parameters
        i_cell = param.current_with_time
        other_side = {"left": "right", "right": "left"}[side]

        c_e_flux = pybamm.BoundaryFlux(c_e, side)
        flux_bc = (
            i_cell / pybamm.BoundaryValue(conductivity, side)
        ) - pybamm.BoundaryValue(param.chi(c_e) / c_e, side) * c_e_flux
        self.boundary_conditions.update(
            {
                delta_phi: {
                    side: (flux_bc, "Neumann"),
                    other_side: (pybamm.Scalar(0), "Neumann"),
                },
                c_e: {
                    side: (c_e_flux, "Neumann"),
                    other_side: (pybamm.Scalar(0), "Neumann"),
                },
            }
        )

    def get_variables_capacitance(self, delta_phi, i_e, Domain):
        """ Get variables for the capacitance models in dictionary form. """
        return {
            Domain + " electrode surface potential difference": delta_phi,
            Domain + " electrolyte current density": i_e,
        }

    def set_leading_order_system(self, delta_phi, reactions, domain, i_curr_coll):
        """
        ODE system for leading-order current in the electrolyte, derived from the
        Stefan-Maxwell equations. If self.use_capacitance is True, this adds equations
        to `rhs`. Otherwise, this adds equations to `algebraic`

        Parameters
        ----------
        delta_phi : :class:`pybamm.Variable`
            The potential difference variable
        reactions : dict
            Dictionary of reaction variables
        domain : list of str
            Domain in which to set the system
        """
        param = self.set_of_parameters

        if domain == ["negative electrode"]:
            x_n = pybamm.standard_spatial_vars.x_n
            i_e = pybamm.outer(i_curr_coll, x_n / param.l_n)
            j_average = i_curr_coll / param.l_n
            j = reactions["main"]["neg"]["aj"]
            self.initial_conditions.update({delta_phi: param.U_n(param.c_n_init)})
            C_dl = param.C_dl_n
            Domain = "Negative"
        elif domain == ["positive electrode"]:
            x_p = pybamm.standard_spatial_vars.x_p
            i_e = pybamm.outer(i_curr_coll, (1 - x_p) / param.l_p)
            j_average = -i_curr_coll / param.l_p
            j = reactions["main"]["pos"]["aj"]
            self.initial_conditions.update({delta_phi: param.U_p(param.c_p_init)})
            C_dl = param.C_dl_p
            Domain = "Positive"
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

        if self.use_capacitance:
            self.rhs.update({delta_phi: 1 / C_dl * (j_average - j)})
        else:
            self.algebraic.update({delta_phi: j_average - j})
        self.variables.update(self.get_variables_capacitance(delta_phi, i_e, Domain))

    def set_post_processed(self, c_e, eps=None):
        """
        Calculate dimensionless and dimensional variables for the capacitance submodel

        Parameters
        ----------
        c_e : :class:`pybamm.Concatenation`
            The eletrolyte concentration variable
        epsilon : :class:`pybamm.Symbol`, optional
            Porosity. Default is None, in which case param.epsilon is used.
        """
        # import parameters and spatial variables
        param = self.set_of_parameters
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack potential differences
        delta_phi_n = self.variables["Negative electrode surface potential difference"]
        delta_phi_p = self.variables["Positive electrode surface potential difference"]
        # Unpack and combine currents
        i_e_n = self.variables["Negative electrolyte current density"]
        i_e_s = pybamm.Broadcast(i_cell, ["separator"])
        i_e_p = self.variables["Positive electrolyte current density"]
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)
        i_s_n = i_cell - i_e_n
        i_s_p = i_cell - i_e_p

        c_e_n, c_e_s, c_e_p = c_e.orphans
        if eps is None:
            eps = param.epsilon
        eps_n, eps_s, eps_p = eps.orphans

        # Negative electrode potential
        solid_conductivity_n = param.sigma_n * (1 - eps_n) ** param.b
        phi_s_n = -pybamm.IndefiniteIntegral(i_s_n / solid_conductivity_n, x_n)
        # Separator electrolyte potential
        phi_e_n = phi_s_n - delta_phi_n
        chi_e_s = param.chi(c_e_s)
        kappa_s_eff = param.kappa_e(c_e_s) * (eps_s ** param.b)
        phi_e_s = pybamm.boundary_value(phi_e_n, "right") + pybamm.IndefiniteIntegral(
            chi_e_s / c_e_s * pybamm.grad(c_e_s) - param.C_e * i_cell / kappa_s_eff, x_s
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
        self.variables.update(self.get_variables(phi_e, i_e, eta_e_av))
        electrode_current_model = pybamm.electrode.Ohm(param)
        vol_vars = electrode_current_model.get_variables(phi_s_n, phi_s_p, i_s_n, i_s_p)
        self.variables.update(vol_vars)

        # Update boundary conditions (for indefinite integral)
        self.boundary_conditions.update(
            {
                c_e_s: {
                    "left": (pybamm.BoundaryFlux(c_e_s, "left"), "Neumann"),
                    "right": (pybamm.BoundaryFlux(c_e_s, "right"), "Neumann"),
                }
            }
        )
