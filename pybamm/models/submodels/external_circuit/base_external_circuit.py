#
# Base model for the external circuit
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Model to represent the behaviour of the external circuit."""

    def __init__(self, param, options):
        super().__init__(param, options=options)

    def get_fundamental_variables(self):
        Q_Ah = pybamm.standard_variables.Q_Ah
        variables = {"Discharge capacity [A.h]": Q_Ah}
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = pybamm.standard_variables.Q_Wh
            variables.update({"Discharge energy [W.h]": Q_Wh})
        return variables

    def set_initial_conditions(self, variables):
        Q_Ah = variables["Discharge capacity [A.h]"]
        self.initial_conditions[Q_Ah] = pybamm.Scalar(0)
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = variables["Discharge energy [W.h]"]
            self.initial_conditions[Q_Wh] = pybamm.Scalar(0)

    def set_rhs(self, variables):
        # ODE for discharge capacity
        Q_Ah = variables["Discharge capacity [A.h]"]
        I = variables["Current [A]"]

        self.rhs[Q_Ah] = I * self.param.timescale / 3600
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = variables["Discharge energy [W.h]"]
            V = variables["Terminal voltage [V]"]
            self.rhs[Q_Wh] = I * V * self.param.timescale / 3600


class LeadingOrderBaseModel(BaseModel):
    """Model to represent the behaviour of the external circuit, at leading order."""

    def __init__(self, param, options):
        super().__init__(param, options)

    def get_fundamental_variables(self):
        Q_Ah = pybamm.Variable("Leading-order discharge capacity [A.h]")
        variables = {"Discharge capacity [A.h]": Q_Ah}
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = pybamm.Variable("Leading-order discharge energy [W.h]")
            variables.update({"Discharge energy [W.h]": Q_Wh})
        return variables
