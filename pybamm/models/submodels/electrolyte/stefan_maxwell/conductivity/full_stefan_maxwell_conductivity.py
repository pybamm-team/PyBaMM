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
        eps = variables["Porosity"]
        c_e = variables["Electrolyte concentration"]
        phi_e = variables["Electrolyte potential"]

        i_e = (param.kappa_e(c_e, T) * (eps ** param.b) * param.gamma_e / param.C_e) * (
            param.chi(c_e) * (1 + param.Theta * T) * pybamm.grad(c_e) / c_e
            - pybamm.grad(phi_e)
        )

        variables.update(self._get_standard_current_variables(i_e))

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
        T_ref = self.param.T_ref
        self.initial_conditions = {phi_e: -self.param.U_n(self.param.c_n_init, T_ref)}

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        if pybamm.have_idaklu():
            return pybamm.IDAKLUSolver()
        else:
            return pybamm.CasadiSolver()
