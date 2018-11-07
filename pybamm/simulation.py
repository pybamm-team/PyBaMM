class Simulation:
    """
    The simulation class for a battery simulation.

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
    def __init__(self, model, parameters, mesh, name='unnamed'):
        self.model = model
        self.parameters = parameters
        self.mesh = mesh
        self.name = name

    def __str__(self):
        return self.name

    def run(self, solver):
        """
        Run the simulation.

        Parameters
        ----------
        solver : pybamm.solver.Solver() instance
            The algorithm for solving the model defined in self.model
        """
        self.solver_name = solver.name
        self.vars = solver.get_simulation_vars(self)

    def save(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
