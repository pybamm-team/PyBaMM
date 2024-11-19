#
# Variables related to discharge and throughput capacity and energy
#
import pybamm
from .base_external_circuit import BaseModel


class DischargeThroughput(BaseModel):
    """Model calculate discharge and throughput capacity and energy."""

    def build(self):
        Q_Ah = pybamm.Variable("Discharge capacity [A.h]")
        Q_Ah.print_name = "Q_Ah"
        # Throughput capacity (cumulative)
        Qt_Ah = pybamm.Variable("Throughput capacity [A.h]")
        Qt_Ah.print_name = "Qt_Ah"

        self.variables = {
            "Discharge capacity [A.h]": Q_Ah,
            "Throughput capacity [A.h]": Qt_Ah,
        }
        if self.options["calculate discharge energy"] == "true":
            Q_Wh = pybamm.Variable("Discharge energy [W.h]")
            # Throughput energy (cumulative)
            Qt_Wh = pybamm.Variable("Throughput energy [W.h]")
            self.variables.update(
                {
                    "Discharge energy [W.h]": Q_Wh,
                    "Throughput energy [W.h]": Qt_Wh,
                }
            )
        else:
            self.variables.update(
                {
                    "Discharge energy [W.h]": pybamm.Scalar(0),
                    "Throughput energy [W.h]": pybamm.Scalar(0),
                }
            )

        # Set initial conditions
        self.initial_conditions[Q_Ah] = pybamm.Scalar(0)
        self.initial_conditions[Qt_Ah] = pybamm.Scalar(0)
        if self.options["calculate discharge energy"] == "true":
            self.initial_conditions[Q_Wh] = pybamm.Scalar(0)
            self.initial_conditions[Qt_Wh] = pybamm.Scalar(0)

        # ODEs for discharge capacity and throughput capacity
        I = pybamm.CoupledVariable("Current [A]")
        self.coupled_variables.update({"Current [A]": I})
        self.rhs[Q_Ah] = I / 3600  # Returns to zero after a complete cycle
        self.rhs[Qt_Ah] = abs(I) / 3600  # Increases with each cycle
        if self.options["calculate discharge energy"] == "true":
            V = pybamm.CoupledVariable("Voltage [V]")
            self.coupled_variables.update({"Voltage [V]": V})
            self.rhs[Q_Wh] = I * V / 3600  # Returns to zero after a complete cycle
            self.rhs[Qt_Wh] = abs(I * V) / 3600  # Increases with each cycle
