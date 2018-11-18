#
# Shared methods for testing
#
import pybamm


def pdes_io(model):
    y = model.initial_conditions()
    vars = pybamm.Variables(0, y, model, model.mesh)
    dydt = model.pdes_rhs(vars)
    return y, dydt
