#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_conductivity import BaseModel


class Full(BaseModel):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseStefanMaxwellConductivity`
    """

    def __init__(self, param, reactions):
        super().__init__(param, reactions=reactions)

    def get_fundamental_variables(self):
        phi_e = pybamm.standard_variables.phi_e
        phi_e_av = pybamm.x_average(phi_e)

        variables = self._get_standard_potential_variables(phi_e, phi_e_av)
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

        # Get neg and pos currents
        T_n, T_s, T_p = T.orphans
        tor_n, tor_s, tor_p = tor.orphans
        c_e_n, c_e_s, c_e_p = c_e.orphans
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans

        i_e_n = (param.kappa_e(c_e_n, T_n) * tor_n * param.gamma_e / param.C_e) * (
            param.chi(c_e_n) * (1 + param.Theta * T_n) * pybamm.grad(c_e_n) / c_e_n
            - pybamm.grad(phi_e_n)
        )
        i_e_p = (param.kappa_e(c_e_p, T_p) * tor_p * param.gamma_e / param.C_e) * (
            param.chi(c_e_p) * (1 + param.Theta * T_p) * pybamm.grad(c_e_p) / c_e_p
            - pybamm.grad(phi_e_p)
        )
        i_typ = self.param.i_typ
        variables.update({
            "Negative electrolyte current density": i_e_n,
            "Positive electrolyte current density": i_e_p,
            "Negative electrolyte current density [A.m-2]": i_e_n * i_typ,
            "Positive electrolyte current density [A.m-2]": i_e_p * i_typ,
        })

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        i_e = variables["Electrolyte current density"]
        sum_j = sum(
            pybamm.Concatenation(
                variables[reaction["Negative"]["aj"]],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                variables[reaction["Positive"]["aj"]],
            )
            for reaction in self.reactions.values()
        )

        self.algebraic = {phi_e: pybamm.div(i_e) - sum_j}

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        T_init = self.param.T_init
        self.initial_conditions = {phi_e: -self.param.U_n(self.param.c_n_init, T_init)}
