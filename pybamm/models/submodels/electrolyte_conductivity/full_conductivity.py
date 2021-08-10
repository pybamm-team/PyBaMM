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
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        if self.half_cell:
            phi_e_n = None
        else:
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
            param.chi(c_e, T) * (1 + param.Theta * T) * pybamm.grad(c_e) / c_e
            - pybamm.grad(phi_e)
        )

        # Override print_name
        i_e.print_name = "i_e"

        variables.update(self._get_standard_current_variables(i_e))
        variables.update(self._get_electrolyte_overpotentials(variables))

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        i_e = variables["Electrolyte current density"]

        # Get surface area to volume ratio (could be a distribution in x to
        # account for graded electrodes)
        a_p = variables["Positive electrode surface area to volume ratio"]
        if self.half_cell:
            a = pybamm.concatenation(
                pybamm.FullBroadcast(0, "separator", "current collector"), a_p
            )
        else:
            a_n = variables["Negative electrode surface area to volume ratio"]
            a = pybamm.concatenation(
                a_n, pybamm.FullBroadcast(0, "separator", "current collector"), a_p
            )

        # Variable summing all of the interfacial current densities
        sum_j = variables["Sum of interfacial current densities"]

        # Override print_name
        sum_j.print_name = "J"
        a.print_name = "a"

        self.algebraic = {phi_e: pybamm.div(i_e) - a * sum_j}

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        self.initial_conditions = {phi_e: -self.param.U_n_init}
