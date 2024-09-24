#
# Equivalent Circuit Model with split OCV
#
import pybamm


class SplitOCVR(pybamm.BaseModel):
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
        super().__init__(name)

        ######################
        # Variables
        ######################
        # All variables are only time-dependent
        # No domain definition needed

        theta_n = pybamm.Variable("Negative particle stoichiometry")
        theta_p = pybamm.Variable("Positive particle stoichiometry")
        Q = pybamm.Variable("Discharge capacity [A.h]")
        V = pybamm.Variable("Voltage [V]")

        # model is isothermal
        I = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})

        # Capacity equation
        self.rhs[Q] = I / 3600
        self.initial_conditions[Q] = pybamm.Scalar(0)

        # Capacity in each electrode
        Q_n = pybamm.Parameter("Negative electrode capacity [A.h]")
        Q_p = pybamm.Parameter("Positive electrode capacity [A.h]")

        # State of charge electrode equations
        theta_n_0 = pybamm.Parameter("Negative electrode initial stoichiometry")
        theta_p_0 = pybamm.Parameter("Positive electrode initial stoichiometry")
        self.rhs[theta_n] = -I / Q_n / 3600
        self.rhs[theta_p] = I / Q_p / 3600
        self.initial_conditions[theta_n] = theta_n_0
        self.initial_conditions[theta_p] = theta_p_0

        # Resistance for IR expression
        R = pybamm.Parameter("Ohmic resistance [Ohm]")

        # Open-circuit potential for each electrode
        Un = pybamm.FunctionParameter(
            "Negative electrode OCP [V]", {"Negative particle stoichiometry": theta_n}
        )
        Up = pybamm.FunctionParameter(
            "Positive electrode OCP [V]", {"Positive particle stoichiometry": theta_p}
        )

        # Voltage expression
        V = Up - Un - I * R

        # Parameters for Voltage cutoff
        voltage_high_cut = pybamm.Parameter("Upper voltage cut-off [V]")
        voltage_low_cut = pybamm.Parameter("Lower voltage cut-off [V]")

        self.variables = {
            "Negative particle stoichiometry": theta_n,
            "Positive particle stoichiometry": theta_p,
            "Current [A]": I,
            "Discharge capacity [A.h]": Q,
            "Voltage [V]": V,
            "Times [s]": pybamm.t,
            "Positive electrode OCP [V]": Up,
            "Negative electrode OCP [V]": Un,
            "Current function [A]": I,
        }

        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", voltage_high_cut - V),
            pybamm.Event("Maximum Negative Electrode stoichiometry", 0.999 - theta_n),
            pybamm.Event("Maximum Positive Electrode stoichiometry", 0.999 - theta_p),
            pybamm.Event("Minimum Negative Electrode stoichiometry", theta_n - 0.0001),
            pybamm.Event("Minimum Positive Electrode stoichiometry", theta_p - 0.0001),
        ]

    @property
    def default_quick_plot_variables(self):
        return [
            "Voltage [V]",
            ["Negative particle stoichiometry", "Positive particle stoichiometry"],
            "Negative electrode OCP [V]",
            "Positive electrode OCP [V]",
            "Current [A]",
        ]
