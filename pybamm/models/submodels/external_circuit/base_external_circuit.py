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
        Q_Wh = pybamm.standard_variables.Q_Wh
        Qt_Wh = pybamm.standard_variables.Qt_Wh
        Qt_Ah = pybamm.standard_variables.Qt_Ah
        variables = {
            "Discharge capacity [A.h]": Q_Ah,
            "Discharge energy [W.h]": Q_Wh,
            "Throughput energy [W.h]": Qt_Wh,
            "Throughput capacity [A.h]": Qt_Ah,
        }
        return variables

    def set_initial_conditions(self, variables):
        for name in [
            "Discharge capacity [A.h]",
            "Discharge energy [W.h]",
            "Throughput energy [W.h]",
            "Throughput capacity [A.h]",
        ]:
            var = variables[name]
            self.initial_conditions[var] = pybamm.Scalar(0)

    def set_rhs(self, variables):
        # ODEs for discharge/throughput capacity/energy
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        for name, integrand in [
            ("Discharge capacity [A.h]", I),
            ("Throughput capacity [A.h]", abs(I)),
            ("Discharge energy [W.h]", I * V),
            ("Throughput energy [W.h]", abs(I * V)),
        ]:
            var = variables[name]
            self.rhs[var] = integrand * self.param.timescale / 3600
