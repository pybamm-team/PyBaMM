#
# Class for electrolyte diffusion employing stefan-maxwell
#
import pybamm

from .base_stefan_maxwell_diffusion import BaseModel


class FullModel(BaseModel):
    """Class for conservation of mass in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseStefanMaxwellDiffusion`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel which can be stated independent of
        variables stated in other submodels
        """

        c_e = pybamm.standard_variables.c_e
        c_e_av = pybamm.average(c_e)

        variables = self._get_standard_concentration_variables(c_e, c_e_av)

        return variables

    def get_coupled_variables(self, variables):

        eps = variables["Porosity"]
        c_e = variables["Electrolyte concentration"]
        i_e = variables["Electrolyte current density"]
        v_box = variables["Volume-averaged velocity"]

        param = self.param

        N_e_diffusion = -(eps ** param.b) * param.D_e(c_e) * pybamm.grad(c_e)
        N_e_migration = (param.C_e * param.t_plus) / param.gamma_e * i_e
        N_e_convection = c_e * v_box

        N_e = N_e_diffusion + N_e_migration + N_e_convection

        variables.update(self._get_standard_flux_variables(N_e))

        return variables

    def set_rhs(self, variables):

        param = self.param

        eps = variables["Porosity"]
        deps_dt = variables["Porosity change"]
        c_e = variables["Electrolyte concentration"]
        N_e = variables["Electrolyte flux"]
        i_e = variables["Electrolyte current density"]

        # TODO: check lead acid version in new form
        # source_terms = param.s / param.gamma_e * j
        source_term = ((param.s - param.t_plus) / param.gamma_e) * pybamm.div(i_e)

        self.rhs = {
            c_e: (1 / eps)
            * (-pybamm.div(N_e) / param.C_e + source_term - c_e * deps_dt)
        }

    def set_boundary_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.boundary_conditions = {
            c_e: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):

        c_e = variables["Electrolyte concentration"]

        self.initial_conditions = {c_e: self.param.c_e_init}

