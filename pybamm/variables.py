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

    def average(self, param, mesh):
        """Average variable attributes over the relevant (sub)domain.

        Parameters
        ----------
        param : pybamm.parameters.Parameters() instance
            The parameters of the simulation.
        mesh : pybamm.mesh.Mesh() instance
            The simulation mesh.

        """
        old_keys = list(self.__dict__.keys())
        for attr in old_keys:
            if attr in ['c']:
                avg = np.dot(self.__dict__[attr], mesh.dx)
                self.__dict__[attr + '_avg'] = avg
            elif attr[-1] == 'n':
                # Negative
                var_n = self.__dict__[attr[:-1]][:mesh.nn-1]
                avg_n = np.sum(var_n) * mesh.dxn / param.ln
                self.__dict__[attr[:-1] + 'n_avg'] = avg_n
                
                # Positive
                var_p = self.__dict__[attr[:-1]][mesh.nn+mesh.ns:]
                avg_p = np.sum(var_p) * mesh.dxp / param.lp
                self.__dict__[attr[:-1] + 'p_avg'] = avg_p
