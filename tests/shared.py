#
# Shared methods for testing
#
import pybamm


def pdes_io(model):
    y = model.initial_conditions()
    vars = pybamm.Variables(model, model.mesh)
    vars.update(0, y)
    dydt = model.pdes_rhs(vars)
    return y, dydt
