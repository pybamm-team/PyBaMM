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

        i_e = (param.kappa_e(c_e, T) * tor * param.gamma_e / param.C_e) * (
            param.chi(c_e) * (1 + param.Theta * T) * pybamm.grad(c_e) / c_e
            - pybamm.grad(phi_e)
        )

        variables.update(self._get_standard_current_variables(i_e))

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        i_e = variables["Electrolyte current density"]

        # Variable summing all of the interfacial current densities
        sum_j = variables["Sum of interfacial current densities"]

        self.algebraic = {phi_e: pybamm.div(i_e) - sum_j}

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        T_init = self.param.T_init
        self.initial_conditions = {
            phi_e: -self.param.U_n(self.param.c_n_init(0), T_init)
        }
