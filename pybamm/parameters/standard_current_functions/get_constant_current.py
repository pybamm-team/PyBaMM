#
# Constant current "function"
#
import pybamm


class GetConstantCurrent(pybamm.GetCurrent):
    """
    Sets a constant input current for a simulation.

    Parameters
    ----------
    current : :class:`pybamm.Symbol` or float
        The size of the current in Amperes.

    **Extends:"": :class:`pybamm.GetCurrent`
    """

    def __init__(self, current=pybamm.electrical_parameters.I_typ):
        self.parameters = {"Current [A]": current}
        self.parameters_eval = {"Current [A]": current}

    def __str__(self):
        return "Constant current"

    def __call__(self, t):
        return self.parameters_eval["Current [A]"]
