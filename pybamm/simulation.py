#
# Simulation class for a battery model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import pickle
import os


class Simulation(object):
    """
    The simulation class for a battery model.

    Parameters
    ---------
    model : pybamm.models.(modelname).(ModelName)() instance
        The model to be used for the simulation. (modelname) and (ModelName)
        refer to a module and class to be chosen.
    param : :class:`pybamm.BaseParameterValues.Parameters` instance
        The parameters to be used for the simulation.
    mesh : :class:`pybamm.mesh.Mesh` instance
        The mesh to be used for the simulation.
    solver : :class:`pybamm.solver.Solver` instance
        The algorithm for solving the model.
    name : string, optional
        The simulation name.

    """

    def __init__(self, model, param=None, mesh=None, solver=None, name="unnamed"):
        # Defaults
        if param is None:
            param = pybamm.BaseParameterValues()
        if mesh is None:
            mesh = pybamm.Mesh(param)
        if solver is None:
            solver = pybamm.Solver()

        # Assign attributes
        self.model = model
        self.param = param
        self.mesh = mesh
        self.solver = solver
        self.name = name

        # Initialise simulation to prepare for solving
        # Set mesh dependent parameters
        self.param.set_mesh(self.mesh)

        # Create operators from solver
        self.operators = self.solver.operators(self.mesh)

        # Assign param, operators and mesh as model attributes
        self.model.set_simulation(self.param, self.operators, self.mesh)

    def __str__(self):
        return self.name

    def run(self, use_force=False):
        """
        Run the simulation.

        Parameters
        ----------
        use_force : boolean, optional
            If False (default), use a stored solution (if it exists). Otherwise, run the
            model.
        """
        # if not use_force and os.path.isfile(self.filename):
        #     self.load()
        # else:
        self.vars = self.solver.get_simulation_vars(self)
        # self.save()

    def average(self):
        """Average simulation variable attributes over the relevant (sub)domain."""
        self.vars.average()

    @property
    def filename_io(self):
        path = os.path.join("out", "simulations")
        filename = "model_{}_solver_{}".format(
            pickle.dumps(self.model), pickle.dumps(self.solver)
        )
        return os.path.join(path, filename)

    def load(self):
        """Load saved simulation if it exists."""
        with open(self.filename_io, "rb") as input:
            return pickle.load(input)

    def save(self):
        """Save simulation."""
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
