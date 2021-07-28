#
# Base model for the external circuit
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Model to represent the behaviour of the external circuit. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        Q = pybamm.standard_variables.Q
        Wh = pybamm.standard_variables.Wh
        variables = {"Discharge capacity [A.h]": Q,
                     "Discharge energy [W.h]": Wh}
        return variables

    def set_initial_conditions(self, variables):
        Q = variables["Discharge capacity [A.h]"]
        Wh = variables["Discharge energy [W.h]"]
        self.initial_conditions[Q] = pybamm.Scalar(0)
        self.initial_conditions[Wh] = pybamm.Scalar(0)

    def set_rhs(self, variables):
        # ODE for discharge capacity
        Q = variables["Discharge capacity [A.h]"]
        Wh = variables["Discharge energy [W.h]"]
        I = variables["Current [A]"]
        V = variables['Terminal voltage [V]']
        self.rhs[Q] = I * self.param.timescale / 3600
        self.rhs[Wh] = I * V * self.param.timescale / 3600


class LeadingOrderBaseModel(BaseModel):
    """Model to represent the behaviour of the external circuit, at leading order. """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        Q = pybamm.Variable("Leading-order discharge capacity [A.h]")
        variables = {"Discharge capacity [A.h]": Q}
        return variables
