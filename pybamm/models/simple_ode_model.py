#
# A model consisting of only ODEs
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class SimpleODEModel(pybamm.BaseModel):
    """A model consisting of only ODEs.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables

    """

    def __init__(self):
        super().__init__()
        # Create variables: domain is explicitly empty since these variables are only
        # functions of time
        a = pybamm.Variable("a", domain=[])
        b = pybamm.Variable("b", domain=[])
        c = pybamm.Variable("c", domain=[])

        # Simple ODEs
        self.rhs = {a: pybamm.Scalar(2), b: pybamm.Scalar(0), c: -c}

        # Simple initial conditions
        self.initial_conditions = {
            a: pybamm.Scalar(0),
            b: pybamm.Scalar(1),
            c: pybamm.Scalar(1),
        }
        # no boundary conditions for an ODE model
        # Broadcast some of the variables
        self.variables = {
            "a": a,
            "b": pybamm.Broadcast(b, ["whole cell"]),
            "c": pybamm.Broadcast(c, ["negative electrode", "separator"]),
        }

        # Overwrite default solver for faster solution
        # self.default_solver = pybamm.ScipySolver(method="BDF")
