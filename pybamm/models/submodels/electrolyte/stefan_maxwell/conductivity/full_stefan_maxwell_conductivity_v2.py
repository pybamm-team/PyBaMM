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
        T_n, T_s, T_p = T.orphans
        tor = variables["Electrolyte tortuosity"]
        tor_n, tor_s, tor_p = tor.orphans
        c_e = variables["Electrolyte concentration"]
        c_e_n, c_e_s, c_e_p = c_e.orphans
        phi_e = variables["Electrolyte potential"]
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans

        i_e_n = (param.kappa_e(c_e_n, T_n) * tor_n * param.gamma_e / param.C_e) * (
            param.chi(c_e_n) * (1 + param.Theta * T_n) * pybamm.grad(c_e_n) / c_e_n
            - pybamm.grad(phi_e_n)
        )
        i_e_s = (param.kappa_e(c_e_s, T_s) * tor_s * param.gamma_e / param.C_e) * (
            param.chi(c_e_s) * (1 + param.Theta * T_s) * pybamm.grad(c_e_s) / c_e_s
            - pybamm.grad(phi_e_s)
        )
        i_e_p = (param.kappa_e(c_e_p, T_p) * tor_p * param.gamma_e / param.C_e) * (
            param.chi(c_e_p) * (1 + param.Theta * T_p) * pybamm.grad(c_e_p) / c_e_p
            - pybamm.grad(phi_e_p)
        )

        i_e = pybamm.Concatenation(i_e_n, i_e_s, i_e_p)

        variables.update(self._get_standard_current_variables(i_e))

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        i_e = variables["Electrolyte current density"]
        i_e_n, i_e_s, i_e_p = i_e.orphans

        sum_j = sum(
            pybamm.Concatenation(
                variables[reaction["Negative"]["aj"]],
                pybamm.FullBroadcast(0, "separator", "current collector"),
                variables[reaction["Positive"]["aj"]],
            )
            for reaction in self.reactions.values()
        )

        self.algebraic = {
            phi_e_n: pybamm.div(i_e_n) - sum(variables[reaction["Negative"]["aj"]] for reaction in self.reactions.values()),
            phi_e_s: pybamm.div(i_e_s),
            phi_e_p: pybamm.div(i_e_p) - sum(variables[reaction["Positive"]["aj"]] for reaction in self.reactions.values()),
        }

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        T_init = self.param.T_init
        self.initial_conditions = {phi_e: -self.param.U_n(self.param.c_n_init, T_init)}

    def set_boundary_conditions(self, variables):
        T = variables["Cell temperature"]
        T_n, T_s, T_p = T.orphans
        c_e = variables["Electrolyte concentration"]
        c_e_n, c_e_s, c_e_p = c_e.orphans
        phi_e = variables["Electrolyte potential"]
        phi_e_n, phi_e_s, phi_e_p = phi_e.orphans
        tor = variables["Electrolyte tortuosity"]
        tor_n, tor_s, tor_p = tor.orphans
        i_cc = variables["Current collector current density"]

        param = self.param
        #neg_rbc = (pybamm.BoundaryValue(param.kappa_e(c_e_s, T_s), "left") / pybamm.BoundaryValue(param.kappa_e(c_e_n, T_n), "right")) * pybamm.BoundaryGradient(phi_e_s, "left")
        #sep_lbc = pybamm.BoundaryValue(phi_e_n, "right")
        #sep_rbc = (pybamm.BoundaryValue(param.kappa_e(c_e_p, T_p), "left") / pybamm.BoundaryValue(param.kappa_e(c_e_s, T_s), "right")) * pybamm.BoundaryGradient(phi_e_p, "left")
        #pos_lbc = pybamm.BoundaryValue(phi_e_s, "right")

        neg_rbc = pybamm.BoundaryValue(param.chi(c_e_n) * (1 + param.Theta * T_n) / c_e_n, "right") * pybamm.BoundaryGradient(c_e_n, "right") - i_cc / (pybamm.BoundaryValue((param.kappa_e(c_e_n, T_n) * tor_n * param.gamma_e / param.C_e), "right"))
        sep_lbc = pybamm.BoundaryValue(phi_e_n, "right")
        sep_rbc = pybamm.BoundaryValue(phi_e_p, "left")
        pos_lbc = pybamm.BoundaryValue(param.chi(c_e_p) * (1 + param.Theta * T_p) / c_e_p, "left") * pybamm.BoundaryGradient(c_e_p, "left") - i_cc / (pybamm.BoundaryValue((param.kappa_e(c_e_p, T_p) * tor_p * param.gamma_e / param.C_e), "left"))

        self.boundary_conditions = {
            phi_e_n: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (neg_rbc, "Neumann"),
            },
            phi_e_s: {
                "left": (sep_lbc, "Dirichlet"),
                "right": (sep_rbc, "Dirichlet"),
            },
            phi_e_p: {
                "left": (pos_lbc, "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
        }
