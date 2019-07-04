#
# Allow a user-defined current function
#
import pybamm


class GetUserCurrent(pybamm.GetCurrent):
    def __init__(self, function, **kwargs):
        # Parameters which may need to be processed
        self.parameters = kwargs
        self.parameters_eval = kwargs

        self.function = function

    def __call__(self, t):

        return self.function(t, **self.parameters_eval)
