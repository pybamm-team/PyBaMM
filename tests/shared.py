#
# Shared methods and classes for testing
#
import pybamm


class VarsForTesting(object):
    def __init__(self, t=None, c=None, e=None):
        self.t = t
        self.c = c
        self.e = e


def pdes_io(model):
    y = model.initial_conditions()
    vars = pybamm.Variables(model, model.mesh)
    vars.update(0, y)
    dydt = model.pdes_rhs(vars)
    return y, dydt
