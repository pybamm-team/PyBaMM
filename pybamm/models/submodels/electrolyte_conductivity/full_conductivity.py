#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_electrolyte_conductivity import BaseElectrolyteConductivity


class Full(BaseElectrolyteConductivity):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        phi_e_n = pybamm.standard_variables.phi_e_n
        phi_e_s = pybamm.standard_variables.phi_e_s
        phi_e_p = pybamm.standard_variables.phi_e_p

        variables = self._get_standard_potential_variables(phi_e_n, phi_e_s, phi_e_p)
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        T = variables["Cell temperature"]
        tor = variables["Electrolyte tortuosity"]
        c_e = variables["Electrolyte concentration"]
        phi_e = variables["Electrolyte potential"]

        c_e_n, c_e_s, c_e_p = c_e.orphans
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        T_n, T_s, T_p = T.orphans

        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e, T) * (1 + param.Theta * T) * pybamm.grad(c_e) / c_e
            - pybamm.grad(phi_e)
        )

        # concentration overpotential
        indef_integral_n = pybamm.IndefiniteIntegral(
            param.chi(c_e_n, T_n) * (1 + param.Theta * T_n) * pybamm.grad(c_e_n) / c_e_n, pybamm.standard_spatial_vars.x_n
        )
        indef_integral_s = pybamm.IndefiniteIntegral(
            param.chi(c_e_s, T_s) * (1 + param.Theta * T_s) * pybamm.grad(c_e_s) / c_e_s, pybamm.standard_spatial_vars.x_s
        )
        indef_integral_p = pybamm.IndefiniteIntegral(
            param.chi(c_e_p, T_p) * (1 + param.Theta * T_p) * pybamm.grad(c_e_p) / c_e_p, pybamm.standard_spatial_vars.x_p
        )

        integral_n = indef_integral_n
        integral_s = indef_integral_s + pybamm.boundary_value(integral_n, "right")
        integral_p = indef_integral_p + pybamm.boundary_value(integral_s, "right")

        eta_c_av = pybamm.x_average(integral_p) - pybamm.x_average(integral_n)

        delta_phi_e_av = (
            pybamm.x_average(phi_e_p) - pybamm.x_average(phi_e_n) - eta_c_av
        )

        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        i_e = variables["Electrolyte current density"]

        # Get surface area to volume ratio (could be a distribution in x to
        # account for graded electrodes)
        a_n = variables["Negative electrode surface area to volume ratio"]
        a_p = variables["Positive electrode surface area to volume ratio"]
        a = pybamm.Concatenation(
            a_n, pybamm.FullBroadcast(0, "separator", "current collector"), a_p
        )

        # Variable summing all of the interfacial current densities
        sum_j = variables["Sum of interfacial current densities"]

        self.algebraic = {phi_e: pybamm.div(i_e) - a * sum_j}

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        T_init = self.param.T_init
        self.initial_conditions = {
            phi_e: -self.param.U_n(self.param.c_n_init(0), T_init)
        }
