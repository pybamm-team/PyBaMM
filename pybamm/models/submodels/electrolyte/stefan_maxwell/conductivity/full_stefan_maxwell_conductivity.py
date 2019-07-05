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


    **Extends:** :class:`pybamm.BaseStefanMaxwellConductivity`
    """

    def __init__(self, param, domain=None):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        phi_e = pybamm.standard_variables.phi_e
        phi_e_av = pybamm.average(phi_e)

        variables = self._get_standard_potential_variables(phi_e, phi_e_av)
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        eps = variables["Porosity"]
        c_e = variables["Electrolyte concentration"]
        phi_e = variables["Electrolyte potential"]

        i_e = (param.kappa_e(c_e) * (eps ** param.b) * param.gamma_e / param.C_e) * (
            param.chi(c_e) * pybamm.grad(c_e) / c_e - pybamm.grad(phi_e)
        )

        variables.update(self._get_standard_current_variables(i_e))

        return variables

    def set_algebraic(self, variables):
        phi_e = variables["Electrolyte potential"]
        i_e = variables["Electrolyte current density"]
        j = variables["Interfacial current density"]

        self.algebraic = {phi_e: pybamm.div(i_e) - j}

    def set_boundary_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]

        self.boundary_conditions = {
            phi_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        phi_e = variables["Electrolyte potential"]
        self.initial_conditions = {phi_e: -self.param.U_n(self.param.c_n_init)}

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
