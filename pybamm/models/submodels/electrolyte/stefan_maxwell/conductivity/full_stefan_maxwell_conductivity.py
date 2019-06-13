#
# Class for electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_conductivity import BaseModel


class FullModel(BaseModel):
    """Class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseStefanMaxwellConductivity`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel which can be stated independent of
        variables stated in other submodels
        """

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

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]
        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)

        self.algebraic = {phi_e: pybamm.div(i_e) - j}

    def set_boundary_conditions(self, variables):

        phi_e = variables["Electrolyte potential"]

        self.boundary_conditions = {
            phi_e: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }

    def set_initial_conditions(self, variables):
        """Pseudo initial conditions for DAE solver (initial guess)"""
        phi_e = variables["Electrolyte potential"]
        self.initial_conditions = {phi_e: -self.param.U_n(self.param.c_n_init)}

