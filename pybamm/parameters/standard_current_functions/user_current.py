#
# Allow a user-defined current function
#
import pybamm


class UserCurrent(pybamm.BaseCurrent):
    """
    Sets a user-defined function as the input current for a simulation.

    Parameters
    ----------
    function : method
        The method which returns the current (in Amperes) as a function of time
        (in seconds). The first argument of function must be time, followed by
        any keyword arguments, i.e. function(t, **kwargs).
    **kwargs : Any keyword arguments required by function.

    **Extends:"": :class:`pybamm.BaseCurrent`
    """

    def __init__(self, function, **kwargs):
        self.parameters = kwargs
        self.parameters_eval = kwargs
        self.function = function

    def __str__(self):
        return "User defined current"

    def __call__(self, t):
        return self.function(t, **self.parameters_eval)
