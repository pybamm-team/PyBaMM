#
# Base model for the external circuit
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Model to represent the behaviour of the external circuit. """

    def __init__(self, param):
        super().__init__(param)

    def _get_current_variables(self, i_cell):
        param = self.param
        I = i_cell * abs(param.I_typ)
        i_cell_dim = I / (param.n_electrodes_parallel * param.A_cc)

        variables = {
            "Total current density": i_cell,
            "Total current density [A.m-2]": i_cell_dim,
            "Current [A]": I,
            "C-rate": I / param.Q,
        }

        return variables

    def get_fundamental_variables(self):
        Q = pybamm.Variable("Discharge capacity [A.h]")
        variables = {"Discharge capacity [A.h]": Q}
        return variables

    def set_initial_conditions(self, variables):
        Q = variables["Discharge capacity [A.h]"]
        self.initial_conditions[Q] = pybamm.Scalar(0)

    def set_rhs(self, variables):
        # ODE for discharge capacity
        Q = variables["Discharge capacity [A.h]"]
        I = variables["Current [A]"]
        self.rhs[Q] = I * self.param.timescale / 3600


class LeadingOrderBaseModel(BaseModel):
    """Model to represent the behaviour of the external circuit, at leading order. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        Q = pybamm.Variable("Leading-order discharge capacity [A.h]")
        variables = {"Discharge capacity [A.h]": Q}
        return variables

