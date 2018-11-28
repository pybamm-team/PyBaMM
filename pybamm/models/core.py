#
# Core model class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class BaseModel(object):
    """Base model class for other models to extend."""

    def __init__(self):
        self.name = "Base Model"
        self.param = pybamm.Parameters()
        self.mesh = pybamm.Mesh()

    def __str__(self):
        return self.name

    def domains(self):
        """The domain(s) in which the model holds."""
        return set([domain for variable, domain in self.variables])

    def initial_conditions(self):
        """Calculate the initial conditions for the simulation.

        Returns
        -------
        y0 : array_like
            A concatenated vector of all the initial conditions.

        """
        raise NotImplementedError

    def pdes_rhs(self, vars):
        """Calculates the spatial derivates of the spatial terms in the PDEs
           and returns the right-hand side to be used by the ODE solver
           (Method of Lines).

        Parameters
        ----------
        vars : pybamm.variables.Variables() instance
            The variables of the model.

        Returns
        -------
        dydt : array_like
            A concatenated vector of all the derivatives.

        """
        raise NotImplementedError
