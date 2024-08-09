#
# Equivalent Circuit Model with split OCV
#
import pybamm
from .base_lithium_ion_model import BaseModel


class ECMsplitOCV(BaseModel):
    """Basic Equivalent Circuit Model that uses two OCV functions
    for each electrode from the OCV function from Lithium ion parameter sets.
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
        T = param.T_init
        I = param.current_with_time

        # Capacity equation
        self.rhs[Q] = I / 3600
        self.initial_conditions[Q] = pybamm.Scalar(0)

        # Capacity in each electrode
        # TODO specify capcity for negative and positive electrodes
        # may be user-defined
        q_n = 1 # Ah
        q_p = 1 # Ah
        Qn = q_n
        Qp = q_p

        # State of charge electrode equations
        self.rhs[c_n] = - I / Qn / 3600
        self.rhs[c_p] = I / Qp / 3600
        self.initial_conditions[c_n] = param.n.prim.c_init_av / param.n.prim.c_max
        self.initial_conditions[c_p] = param.p.prim.c_init_av / param.p.prim.c_max

        # OCV's for the electrodes
        Un = param.n.prim.U(c_n, T)
        Up = param.p.prim.U(c_p, T)

        # IR resistance, hard coded for now
        IR = 0.1
        V = Up - Un - IR

        self.variables = {
            "Negative particle SOC": c_n,
            "Positive particle SOC": c_p,
            "Current [A]": I,
            "Discharge capacity [A.h]": Q,
            "Voltage [V]": V,
            "Times [s]": pybamm.t,
            "Positive electrode potential [V]": Up,
            "Negative electrode potential [V]": Un,
            "Current variable [A]": I,
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
