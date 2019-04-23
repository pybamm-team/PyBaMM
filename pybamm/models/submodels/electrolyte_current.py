#
# Equation classes for the electrolyte current
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class MacInnesStefanMaxwell(pybamm.SubModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
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


class MacInnesStefanMaxwellCapacitance(pybamm.LeadAcidBaseModel):
    """MacInnes equation for the current in the electrolyte, derived from the
    Stefan-Maxwell equations, with capacitance effects included. The MacInnes equation
    is rearranged to account for both solid and electrolyte potentials

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        The electrolyte concentration
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (can be Variable or Parameter)
    Delta_phi : :class:`pybamm.Symbol`
        The difference between the electric potential in the electrolyte and the
        electric potential in the solid (Delta_phi = phi_s - phi_e)
    j : :class:`pybamm.Symbol`
        The interfacial current density at the electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, eps, Delta_phi, j, param):
        super().__init__()
        current = pybamm.standard_parameters.current_with_time

        # ode model only
        self.algebraic = {}
        i_e = (
            param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_hat_e
        ) * (param.chi(c_e) * pybamm.grad(c_e) / c_e + pybamm.grad(Delta_phi))

        # different bounday conditions in each electrode
        if Delta_phi.domain == ["negative electrode"]:
            self.rhs = {Delta_phi: 1 / param.gamma_dl_n * (pybamm.div(i_e) - j)}
            self.boundary_conditions = {Delta_phi: {"left": 0}, i_s_n: {"right": 0}}
            self.boundary_conditions = {i_e: {"left": 0, "right": current}}
            self.initial_conditions = {Delta_phi: param.U_n(param.c_e_init)}
            self.variables = {
                "Negative electrode potential difference": Delta_phi,
                "Negative electrode electrolyte current": i_e,
            }
        elif Delta_phi.domain == ["positive electrode"]:
            self.rhs = {Delta_phi: 1 / param.gamma_dl_p * (pybamm.div(i_e) - j)}
            self.boundary_conditions = {i_e: {"left": current, "right": 0}}
            self.initial_conditions = {Delta_phi: param.U_p(param.c_e_init)}
            self.variables = {
                "Positive electrode potential difference": Delta_phi,
                "Positive electrode electrolyte current": i_e,
            }
        # for whole cell domain call both electrode models and ignore separator
        elif Delta_phi.domain == [
            "negative electrode",
            "separator",
            "positive electrode",
        ]:
            c_e_n, c_e_s, c_e_p = c_e.orphans
            Delta_phi_n, Delta_phi_s, Delta_phi_p = Delta_phi.orphans
            eps_n, eps_s, eps_p = eps.orphans
            j_n, j_s, j_p = j.orphans
            neg_model = MacInnesStefanMaxwellCapacitance(
                c_e_n, Delta_phi_n, eps_n, j_n, param
            )
            pos_model = MacInnesStefanMaxwellCapacitance(
                c_e_p, Delta_phi_p, eps_p, j_p, param
            )
            self.update(neg_model, pos_model)
            # Voltage variable
            voltage = pybamm.BoundaryValue(phi_s, "right") - pybamm.BoundaryValue(
                phi_s, "left"
            )
            self.variables.update({"Voltage": voltage})
        else:
            raise pybamm.DomainError(
                "domain '{}' not recognised".format(Delta_phi.domain)
            )
