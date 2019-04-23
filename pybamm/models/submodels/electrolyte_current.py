#
# Equation classes for the electrolyte current
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class ElectrolyteCurrentBaseModel(pybamm.SubModel):
    """
    Base model for the potential and current in the electrolyte

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def get_variables(self, phi_e, i_e, eta_c_av, delta_phi_e_av, eta_e_av):
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

        # Set dimensionless and dimensional variables
        return {
            "Electrolyte potential": phi_e,
            "Electrolyte current density": i_e,
            "Average concentration overpotential": eta_c_av,
            "Average electrolyte ohmic losses": delta_phi_e_av,
            "Average electrolyte overpotential": eta_e_av,
            "Electrolyte potential [V]": -param.U_n_ref + pot_scale * phi_e,
            "Electrolyte current density [A m-2]": param.i_typ * i_e,
            "Average concentration overpotential [V]": pot_scale * eta_c_av,
            "Average electrolyte ohmic losses [V]": pot_scale * delta_phi_e_av,
            "Average electrolyte overpotential [V]": pot_scale * eta_e_av,
        }


class MacInnesStefanMaxwell(ElectrolyteCurrentBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`ElectrolyteCurrentBaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_algebraic_system(self, phi_e, variables):
        """
        PDE system for current in the electrolyte, derived from the Stefan-Maxwell
        equations.

        Parameters
        ----------
        phi_e : :class:`pybamm.Concatenation`
            The eletrolyte potential variable
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables
        """
        # Load parameters and spatial variables
        param = self.set_of_parameters
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack variables
        c_e = variables["Electrolyte concentration"]
        j = variables["Interfacial current density"]

        # if porosity is not provided, use the input parameter
        try:
            epsilon = variables["Porosity"]
        except KeyError:
            epsilon = param.epsilon

        # functions
        i_e = (
            param.kappa_e(c_e) * (epsilon ** param.b) * param.gamma_e / param.C_e
        ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e))

        # Equations (algebraic only)
        self.algebraic = {phi_e: pybamm.div(i_e) - j}
        self.boundary_conditions = {i_e: {"left": 0, "right": 0}}
        self.initial_conditions = {phi_e: -param.U_n(param.c_n_init)}
        # no differential equations
        self.rhs = {}

        # Variables
        # eta_c_av and delta_phi_e_av not defined?
        eta_c_av = pybamm.Scalar(0)
        delta_phi_e_av = pybamm.Scalar(0)

        # average elecrolyte overpotential (ohmic + concentration overpotential)
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        phi_e_n_av = pybamm.Integral(phi_e_n, x_n) / param.l_n
        phi_e_p_av = pybamm.Integral(phi_e_p, x_p) / param.l_p
        eta_e_av = phi_e_p_av - phi_e_n_av

        self.variables = self.get_variables(
            phi_e, i_e, eta_c_av, delta_phi_e_av, eta_e_av
        )

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()

    def get_explicit_leading_order(self, variables):
        """
        Provides explicit leading order solution to the electrolyte current conservation
        equation where the constitutive equation is taken to be of Stefan-Maxwell form.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

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

        # unpack variables
        ocp_n = variables["Negative electrode open circuit potential"]
        eta_r_n = variables["Negative reaction overpotential"]

        # get left-most ocp and overpotential
        ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
        eta_r_n_left = pybamm.BoundaryValue(eta_r_n, "left")

        # electrolye potential
        phi_e_const = -ocp_n_left - eta_r_n_left
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

        return self.get_variables(phi_e, i_e, eta_c_av, delta_phi_e_av, eta_e_av)

    def get_explicit_combined(self, variables):
        """
        Provides and explicit combined leading and first order solution to the
        electrolyte current conservation equation where the constitutive equation is
        taken to be of Stefan-Maxwell form. Note that the returned current density is
        only the leading order approximation.

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial vairables
        param = self.set_of_parameters
        l_n = param.l_n
        l_p = param.l_p
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack variables
        c_e = variables["Electrolyte concentration"]
        ocp_n = variables["Negative electrode open circuit potential"]
        eta_r_n = variables["Negative reaction overpotential"]

        # extract c_e components
        c_e_n, c_e_s, c_e_p = c_e.orphans

        # if porosity is not provided, use the input parameter
        try:
            epsilon = variables["Porosity (leading-order)"]
        except KeyError:
            epsilon = param.epsilon
        try:
            c_e_0 = (
                variables["Electrolyte concentration (leading-order)"]
                .orphans[0]
                .orphans[0]
            )
        except KeyError:
            c_e_0 = pybamm.Scalar(1)
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]

        # bulk conductivities (leading order)
        kappa_n = param.kappa_e(c_e_0) * eps_n ** param.b
        kappa_s = param.kappa_e(c_e_0) * eps_s ** param.b
        kappa_p = param.kappa_e(c_e_0) * eps_p ** param.b

        # get left-most ocp and overpotential
        ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
        eta_r_n_left = pybamm.BoundaryValue(eta_r_n, "left")
        c_e_n_left = pybamm.BoundaryValue(c_e_n, "left")

        # electrolyte current (leading-order approximation)
        i_e_n = i_cell * x_n / l_n
        i_e_s = pybamm.Broadcast(i_cell, ["separator"])
        i_e_p = i_cell * (1 - x_p) / l_p
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        # electrolyte potential (combined leading and first order)
        phi_e_const = (
            -ocp_n_left
            - eta_r_n_left
            - (
                2
                * param.C_e
                * (1 - param.t_plus)
                * pybamm.Function(np.log, c_e_n_left / c_e_0)
            )
            + (
                param.C_e
                * i_cell
                / param.gamma_e
                * (-l_n / (2 * kappa_n) + (l_n / kappa_s))
            )
        )
        phi_e_n = phi_e_const + param.C_e * (
            2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_n / c_e_0)
            - (i_cell / param.gamma_e)
            * ((x_n ** 2 - l_n ** 2) / (2 * kappa_n * l_n) + l_n / kappa_s)
        )
        phi_e_s = phi_e_const + param.C_e * (
            2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_s / c_e_0)
            - (i_cell / param.gamma_e) * (x_s / kappa_s)
        )
        phi_e_p = phi_e_const + param.C_e * (
            2 * (1 - param.t_plus) * pybamm.Function(np.log, c_e_p / c_e_0)
            - (i_cell / param.gamma_e)
            * (
                (x_p * (2 - x_p) - l_p ** 2 - 1) / (2 * kappa_p * l_p)
                + (1 - l_p) / kappa_s
            )
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        "Ohmic losses and overpotentials"
        # average electrolyte ohmic losses
        delta_phi_e_av = -(
            param.C_e * i_cell / param.gamma_e / param.kappa_e(c_e_0)
        ) * (
            param.l_n / (3 * eps_n ** param.b)
            + param.l_s / (eps_s ** param.b)
            + param.l_p / (3 * eps_p ** param.b)
        )

        # electrode-averaged electrolye concentrations (combined leading
        # and first order)
        c_e_n_av = pybamm.Integral(c_e_n, x_n) / l_n
        c_e_p_av = pybamm.Integral(c_e_p, x_p) / l_p

        # concentration overpotential (combined leading and first order)
        eta_c_av = 2 * param.C_e * (1 - param.t_plus) * (c_e_p_av - c_e_n_av)
        # electrolyte overpotential
        eta_e_av = eta_c_av + delta_phi_e_av

        # Variables
        return self.get_variables(phi_e, i_e, eta_c_av, delta_phi_e_av, eta_e_av)


class MacInnesCapacitance(ElectrolyteCurrentBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations, with capacitance effects included. The MacInnes equation
    is rearranged to account for both solid and electrolyte potentials

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`ElectrolyteCurrentBaseModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, delta_phi, variables):
        param = self.set_of_parameters
        i_cell = param.current_with_time

        # ode model only
        self.algebraic = {}

        if delta_phi.domain == ["negative electrode"]:
            c_e = variables["Negative electrolyte concentration"]
            try:
                eps = variables["Negative electrode porosity"]
            except KeyError:
                eps = param.epsilon_n
            j = variables["Negative electrode interfacial current density"]

            i_e = (
                param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e
            ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e + pybamm.grad(delta_phi))
            self.rhs = {delta_phi: 1 / param.C_dl_n * (pybamm.div(i_e) - j)}
            self.boundary_conditions = {i_e: {"left": 0, "right": i_cell}}
            self.initial_conditions = {delta_phi: param.U_n(param.c_e_init)}
            self.variables = {
                "Negative electrode potential difference": delta_phi,
                "Negative electrolyte current density": i_e,
            }
        elif delta_phi.domain == ["positive electrode"]:
            c_e = variables["Positive electrolyte concentration"]
            try:
                eps = variables["Positive electrode porosity"]
            except KeyError:
                eps = param.epsilon_p
            j = variables["Positive electrode interfacial current density"]

            i_e = (
                param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e
            ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e + pybamm.grad(delta_phi))
            self.rhs = {delta_phi: 1 / param.C_dl_p * (pybamm.div(i_e) - j)}
            self.boundary_conditions = {i_e: {"left": i_cell, "right": 0}}
            self.initial_conditions = {delta_phi: param.U_p(param.c_e_init)}
            self.variables = {
                "Positive electrode potential difference": delta_phi,
                "Positive electrolyte current density": i_e,
            }
        else:
            raise pybamm.DomainError(
                "domain '{}' not recognised".format(delta_phi.domain)
            )

    def set_leading_order_system(self, delta_phi, variables, domain):
        param = self.set_of_parameters
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        # ode model only
        self.algebraic = {}

        if domain == ["negative electrode"]:
            j = variables["Negative electrode interfacial current density"].orphans[0]

            i_e = i_cell * x_n / param.l_n
            self.rhs = {delta_phi: 1 / param.C_dl_n * (i_cell / param.l_n - j)}
            self.initial_conditions = {delta_phi: param.U_n(param.c_e_init)}
            self.variables = {
                "Negative electrode potential difference": delta_phi,
                "Negative electrolyte current density": i_e,
            }
        elif domain == ["positive electrode"]:
            j = variables["Positive electrode interfacial current density"].orphans[0]

            i_e = i_cell * (1 - x_p) / param.l_p
            self.rhs = {delta_phi: 1 / param.C_dl_p * (-i_cell / param.l_p - j)}
            self.initial_conditions = {delta_phi: param.U_p(param.c_e_init)}
            self.variables = {
                "Positive electrode potential difference": delta_phi,
                "Positive electrolyte current density": i_e,
            }
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))

    def get_post_processed(self, variables):
        """
        Calculate dimensionless and dimensional variables for the capacitance submodel

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial vairables
        param = self.set_of_parameters
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack and combine
        delta_phi_n = variables["Negative electrode potential difference"]
        delta_phi_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        delta_phi_p = variables["Positive electrode potential difference"]
        delta_phi = pybamm.Concatenation(delta_phi_n, delta_phi_s, delta_phi_p)

        i_e_n = variables["Negative electrolyte current density"]
        i_e_s = pybamm.Broadcast(i_cell, ["separator"])  # can we put NaN?
        i_e_p = variables["Positive electrolyte current density"]
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        c_e_n = variables["Negative electrolyte concentration"]
        c_e_s = variables["Separator electrolyte concentration"]
        c_e_p = variables["Positive electrolyte concentration"]

        eps_n = variables["Negative electrode porosity"]
        eps_s = variables["Separator porosity"]
        eps_p = variables["Positive electrode porosity"]

        # Compute potentials
        phi_e_children = [None] * 3
        for i, (c_e, eps, i_e, x) in enumerate(
            [
                (c_e_n, eps_n, i_e_n, x_n),
                (c_e_s, eps_s, i_e_s, x_s),
                (c_e_p, eps_p, i_e_p, x_p),
            ]
        ):
            chi_e = param.chi(c_e)
            kappa_eff = param.kappa_e(c_e) * (eps ** param.b)
            d_phi_e__d_x = chi_e / c_e * pybamm.grad(c_e) - param.C_e * i_e / kappa_eff
            phi_e_children[i] = pybamm.IndefiniteIntegral(d_phi_e__d_x, x)

        # Adjust for boundary conditions and continuity
        phi_e_n, phi_e_s, phi_e_p = phi_e_children
        phi_e_n = phi_e_n + pybamm.BoundaryValue(-delta_phi_n - phi_e_n, "left")
        phi_e_s = (
            phi_e_s
            - pybamm.BoundaryValue(phi_e_s, "left")
            + pybamm.BoundaryValue(phi_e_n, "right")
        )
        phi_e_p = (
            phi_e_p
            - pybamm.BoundaryValue(phi_e_p, "left")
            + pybamm.BoundaryValue(phi_e_s, "right")
        )

        # Concatenate
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Other variables
        # eta_c_av and delta_phi_e_av not defined?
        eta_c_av = pybamm.Scalar(0)
        delta_phi_e_av = pybamm.Scalar(0)

        # average elecrolyte overpotential (ohmic + concentration overpotential)
        phi_e_n_av = pybamm.Integral(phi_e_n, x_n) / param.l_n
        phi_e_p_av = pybamm.Integral(phi_e_p, x_p) / param.l_p
        eta_e_av = phi_e_p_av - phi_e_n_av

        return self.get_variables(phi_e, i_e, eta_c_av, delta_phi_e_av, eta_e_av)

    def get_post_processed_leading_order(self, variables):
        """
        Calculate dimensionless and dimensional variables for the capacitance submodel

        Parameters
        ----------
        variables : dict
            Dictionary of {string: :class:`pybamm.Symbol`}, which can be read to find
            already-calculated variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        # import parameters and spatial vairables
        param = self.set_of_parameters
        i_cell = param.current_with_time
        x_n = pybamm.standard_spatial_vars.x_n
        x_s = pybamm.standard_spatial_vars.x_s
        x_p = pybamm.standard_spatial_vars.x_p

        # Unpack and combine
        delta_phi_n = variables["Negative electrode potential difference"]
        delta_phi_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?
        delta_phi_p = variables["Positive electrode potential difference"]
        delta_phi = pybamm.Concatenation(delta_phi_n, delta_phi_s, delta_phi_p)

        i_e_n = variables["Negative electrolyte current density"]
        i_e_s = pybamm.Broadcast(i_cell, ["separator"])  # can we put NaN?
        i_e_p = variables["Positive electrolyte current density"]
        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        c_e_n = variables["Negative electrolyte concentration"]
        c_e_s = variables["Separator electrolyte concentration"]
        c_e_p = variables["Positive electrolyte concentration"]

        eps_n = variables["Negative electrode porosity"]
        eps_s = variables["Separator porosity"]
        eps_p = variables["Positive electrode porosity"]

        # Compute potentials
        phi_e_children = [None] * 3
        for i, (c_e, eps, i_e, x) in enumerate(
            [
                (c_e_n, eps_n, i_e_n, x_n),
                (c_e_s, eps_s, i_e_s, x_s),
                (c_e_p, eps_p, i_e_p, x_p),
            ]
        ):
            chi_e = param.chi(c_e)
            kappa_eff = param.kappa_e(c_e) * (eps ** param.b)
            d_phi_e__d_x = chi_e / c_e * pybamm.grad(c_e) - param.C_e * i_e / kappa_eff
            phi_e_children[i] = pybamm.IndefiniteIntegral(d_phi_e__d_x, x)

        # Adjust for boundary conditions and continuity
        phi_e_n, phi_e_s, phi_e_p = phi_e_children
        phi_e_n = phi_e_n + pybamm.BoundaryValue(-delta_phi_n - phi_e_n, "left")
        phi_e_s = (
            phi_e_s
            - pybamm.BoundaryValue(phi_e_s, "left")
            + pybamm.BoundaryValue(phi_e_n, "right")
        )
        phi_e_p = (
            phi_e_p
            - pybamm.BoundaryValue(phi_e_p, "left")
            + pybamm.BoundaryValue(phi_e_s, "right")
        )

        # Concatenate
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Other variables
        # eta_c_av and delta_phi_e_av not defined?
        eta_c_av = pybamm.Scalar(0)
        delta_phi_e_av = pybamm.Scalar(0)

        # average elecrolyte overpotential (ohmic + concentration overpotential)
        phi_e_n_av = pybamm.Integral(phi_e_n, x_n) / param.l_n
        phi_e_p_av = pybamm.Integral(phi_e_p, x_p) / param.l_p
        eta_e_av = phi_e_p_av - phi_e_n_av

        return self.get_variables(phi_e, i_e, eta_c_av, delta_phi_e_av, eta_e_av)
