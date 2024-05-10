#
# Variables related to discharge and throughput capacity and energy
#
import pybamm
from .base_external_circuit import BaseModel


class DischargeThroughput(BaseModel):
    """Model calculate discharge and throughput capacity and energy."""

    def get_fundamental_variables(self):
        Q_Ah = pybamm.Variable("Discharge capacity [A.h]")
        Q_Ah.print_name = "Q_Ah"
        # Throughput capacity (cumulative)
        Qt_Ah = pybamm.Variable("Throughput capacity [A.h]")
        Qt_Ah.print_name = "Qt_Ah"

        variables = {
            "Discharge capacity [A.h]": Q_Ah,
            "Throughput capacity [A.h]": Qt_Ah,
        }
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = pybamm.Variable("Discharge energy [W.h]")
            # Throughput energy (cumulative)
            Qt_Wh = pybamm.Variable("Throughput energy [W.h]")
            variables.update(
                {
                    "Discharge energy [W.h]": Q_Wh,
                    "Throughput energy [W.h]": Qt_Wh,
                }
            )
        else:
            variables.update(
                {
                    "Discharge energy [W.h]": pybamm.Scalar(0),
                    "Throughput energy [W.h]": pybamm.Scalar(0),
                }
            )
        return variables

    def set_initial_conditions(self, variables):
        Q_Ah = variables["Discharge capacity [A.h]"]
        Qt_Ah = variables["Throughput capacity [A.h]"]
        self.initial_conditions[Q_Ah] = pybamm.Scalar(0)
        self.initial_conditions[Qt_Ah] = pybamm.Scalar(0)
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = variables["Discharge energy [W.h]"]
            Qt_Wh = variables["Throughput energy [W.h]"]
            self.initial_conditions[Q_Wh] = pybamm.Scalar(0)
            self.initial_conditions[Qt_Wh] = pybamm.Scalar(0)

    def set_rhs(self, variables):
        # ODEs for discharge capacity and throughput capacity
        Q_Ah = variables["Discharge capacity [A.h]"]
        Qt_Ah = variables["Throughput capacity [A.h]"]
        I = variables["Current [A]"]
        self.rhs[Q_Ah] = I / 3600  # Returns to zero after a complete cycle
        self.rhs[Qt_Ah] = abs(I) / 3600  # Increases with each cycle
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = variables["Discharge energy [W.h]"]
            Qt_Wh = variables["Throughput energy [W.h]"]
            V = variables["Voltage [V]"]
            self.rhs[Q_Wh] = I * V / 3600  # Returns to zero after a complete cycle
            self.rhs[Qt_Wh] = abs(I * V) / 3600  # Increases with each cycle
