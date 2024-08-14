#
# Equivalent Circuit Model with split OCV
#
import pybamm
from .base_lithium_ion_model import BaseModel


class ECMsplitOCV(BaseModel):
    """Basic Equivalent Circuit Model that uses two OCV functions
    for each electrode. This model is easily parameterizable with minimal parameters.
    This class differs from the :class: pybamm.equivalent_circuit.Thevenin() due
    to dual OCV functions to make up the voltage from each electrode.

    Parameters
    ----------
    name: str, optional
        The name of the model.
    """

    def __init__(self, name="ECM with split OCV"):
        super().__init__({}, name)
        # TODO citations
        param = self.param

        ######################
        # Variables
        ######################
        # All variables are only time-dependent
        # No domain definition needed

        c_n = pybamm.Variable("Negative particle SOC")
        c_p = pybamm.Variable("Positive particle SOC")
        Q = pybamm.Variable("Discharge capacity [A.h]")
        V = pybamm.Variable("Voltage [V]")

        # model is isothermal
        I = param.current_with_time

        # Capacity equation
        self.rhs[Q] = I / 3600
        self.initial_conditions[Q] = pybamm.Scalar(0)

        # Capacity in each electrode
        Q_n = pybamm.Parameter("Negative electrode capacity [A.h]")
        Q_p = pybamm.Parameter("Positive electrode capacity [A.h]")

        # State of charge electrode equations
        c_n_0 = pybamm.Parameter("Negative electrode initial SOC")
        c_p_0 = pybamm.Parameter("Positive electrode initial SOC")
        self.rhs[c_n] = -I / Q_n / 3600
        self.rhs[c_p] = I / Q_p / 3600
        self.initial_conditions[c_n] = c_n_0
        self.initial_conditions[c_p] = c_p_0

        # Resistance for IR expression
        R = pybamm.Parameter("Ohmic resistance [Ohm]")

        # Open-circuit potential for each electrode
        Un = pybamm.FunctionParameter(
            "Negative electrode OCP [V]", {"Negative particle SOC": c_n}
        )
        Up = pybamm.FunctionParameter(
            "Positive electrode OCP [V]", {"Positive particle SOC": c_p}
        )

        # Voltage expression
        V = Up - Un - I * R

        self.variables = {
            "Negative particle SOC": c_n,
            "Positive particle SOC": c_p,
            "Current [A]": I,
            "Discharge capacity [A.h]": Q,
            "Voltage [V]": V,
            "Times [s]": pybamm.t,
            "Positive electrode potential [V]": Up,
            "Negative electrode potential [V]": Un,
            "Current function [A]": I,
        }

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - V),
            pybamm.Event("Maximum Negative Electrode SOC", 0.999 - c_n),
            pybamm.Event("Maximum Positive Electrode SOC", 0.999 - c_p),
            pybamm.Event("Minimum Negative Electrode SOC", c_n - 0.0001),
            pybamm.Event("Minimum Positive Electrode SOC", c_p - 0.0001),
        ]
