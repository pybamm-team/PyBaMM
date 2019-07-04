#
# Constant current "function"
#
import pybamm


class GetConstantCurrent(pybamm.GetCurrent):
    def __init__(self, current=pybamm.electrical_parameters.I_typ):
        # Parameters which need to be processed
        self.parameters = {"Current [A]": current}
        self.parameters_eval = {"Current [A]": current}

    def __call__(self, t):
        return self.parameters_eval["Current [A]"]
