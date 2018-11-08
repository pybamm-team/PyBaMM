import numpy as np

class Variables:
    """
    Extracts and stores the model variables.

    Parameters
    ----------
    y : array_like
        An array containing variable values to be extracted.
    param : pybamm.parameters.Parameters() instance
        The parameters of the simulation.
    mesh : pybamm.mesh.Mesh() instance
        The simulation mesh.
    """
    def __init__(self, y, param, mesh):
        # Split y
        self.c = y
        # TODO: make model.variables an input of this class
        self.cn, self.cs, self.cp = np.split(
            self.c, np.cumsum([mesh.nn - 1, mesh.ns + 1]))
