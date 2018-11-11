import numpy as np

class Variables:
    """
    Extracts and stores the model variables.

    Parameters
    ----------
    t : float or array_like
        The simulation time.
    y : array_like
        An array containing variable values to be extracted.
    param : pybamm.parameters.Parameters() instance
        The simulation parameters.
    mesh : pybamm.mesh.Mesh() instance
        The simulation mesh.
    """
    def __init__(self, t, y, model, mesh):
        self.t = t
        # Unpack y
        variables = model.variables()
        # Unpack y iteratively
        start = 0
        for var, domain in variables:
            end = start + len(mesh.__dict__[domain])
            self.__dict__[var] = y[start:end]
            start = end
            # Split 'tot' variables further into n, s and p
            if domain == 'xc':
                (self.__dict__[var+'n'],
                 self.__dict__[var+'s'],
                 self.__dict__[var+'p']) = np.split(
                     self.__dict__[var],
                     np.cumsum([mesh.nn - 1, mesh.ns + 1]))

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
                avg = np.dot(self.__dict__[attr].T, mesh.dx)
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
