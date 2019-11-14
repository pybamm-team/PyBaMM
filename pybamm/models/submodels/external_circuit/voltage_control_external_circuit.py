#
# External circuit with voltage control
#
import pybamm
from .base_external_circuit import BaseModel


class VoltageControl(BaseModel):
    """External circuit with voltage control. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        # Current is a variable
        param = self.param
        i_cell = pybamm.Variable("Total current density")
        I = i_cell * abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
        }

        return variables

    def set_initial_conditions(self, variables):
        # Initial condition as a guess for consistent initial conditions
        i_cell = variables["Total current density"]
        self.initial_conditions[i_cell] = self.param.current_with_time

    def set_algebraic(self, variables):
        # Read off the current from the solid current density
        # TODO: investigate whether defining the current differently gives better
        # results (e.g. based on electrolyte current density in the separator)
        # This also needs to be defined in a model-inpedendent way (currently assumes
        # Ohm's law)
        i_cell = variables["Total current density"]
        phi_s_p = variables["Positive electrode potential"]
        tor_p = variables["Positive electrode tortuosity"]
        sigma_eff = self.param.sigma_p * tor_p
        i_s_p_right = -pybamm.boundary_value(
            sigma_eff, "right"
        ) * pybamm.BoundaryGradient(phi_s_p, "right")

        self.algebraic[i_cell] = i_cell - i_s_p_right
