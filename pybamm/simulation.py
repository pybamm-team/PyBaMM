#
# Simulation class for a battery model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals


class Simulation(object):
    """
    The simulation class for a battery model.

    Parameters
    ---------
    model : pybamm.models.(modelname).(ModelName)() instance
        The model to be used for the simulation. (modelname) and (ModelName)
        refer to a module and class to be chosen.

    parameters : pybamm.parameters.Parameters() instance
        The parameters to be used for the simulation.

    mesh : pybamm.mesh.Mesh() instance
        The mesh to be used for the simulation.

    name : string, optional
        The simulation name.

    """

    def __init__(self, model, param, mesh, name="unnamed"):
        self.model = model
        self.param = param
        self.mesh = mesh
        self.name = name

    def __str__(self):
        return self.name

    def initialise(self, solver):
        """Initialise simulation to prepare for solving.

        Parameters
        ----------
        solver : :class:`pybamm.solver.Solver` instance
            The algorithm for solving the model defined in self.model.
        """
        # Set mesh dependent parameters
        self.param.set_mesh_dependent_parameters(self.mesh)

        # Create operators from solver
        self.operators = solver.operators(self.model.domains(), self.mesh)

        # Assign param, operators and mesh as model attributes
        self.model.set_simulation(self.param, self.operators, self.mesh)

    def run(self, solver, use_force=False):
        """
        Run the simulation.

        Parameters
        ----------
        solver : :class:`pybamm.solver.Solver` instance
            The algorithm for solving the model defined in self.model.
        """
        self.solver_name = str(solver)
        self.initialise(solver)
        self.vars = solver.get_simulation_vars(self)

    def average(self):
        """Average simulation variable attributes
        over the relevant (sub)domain."""
        self.vars.average()

    def save(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
