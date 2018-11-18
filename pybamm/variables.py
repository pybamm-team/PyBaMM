#
# Variables of a model or simulation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Variables(object):
    """
    Extracts and stores the model variables.

    Parameters
    ----------
    model : A model class instance
        The simulation model.
    mesh : :class:`pybamm.mesh.Mesh` instance
        The simulation mesh.
    """

    def __init__(self, model, mesh):
        self.model = model
        self.mesh = mesh

    def update(self, t, y):
        """Update variables with a new t and y.
        Note that we can't store variables at each timestep since the
        timestepping is adaptive.

        Parameters
        ----------
        t : float or array_like
            The simulation time.
        y : array_like
            An array containing variable values to be extracted.
        """
        self.t = t
        # Unpack y iteratively
        start = 0
        for var, domain in self.model.variables:
            end = start + len(self.mesh.__dict__[domain])
            self.__dict__[var] = y[start:end]
            start = end
            # Split 'tot' variables further into n, s and p
            if domain == "xc":
                (
                    self.__dict__[var + "n"],
                    self.__dict__[var + "s"],
                    self.__dict__[var + "p"],
                ) = np.split(
                    self.__dict__[var],
                    np.cumsum([self.mesh.nn - 1, self.mesh.ns + 1]),
                )

    def average(self):
        """Average variable attributes over the relevant (sub)domain."""
        param = self.model.param
        mesh = self.mesh
        old_keys = list(self.__dict__.keys())
        for attr in old_keys:
            if attr in ["c"]:
                avg = np.dot(self.__dict__[attr].T, mesh.dx)
                self.__dict__[attr + "_avg"] = avg
            elif attr[-1] == "n":
                # Negative
                var_n = self.__dict__[attr[:-1]][: mesh.nn - 1]
                avg_n = np.sum(var_n) * mesh.dxn / param.ln
                self.__dict__[attr[:-1] + "n_avg"] = avg_n

                # Separator
                var_s = self.__dict__[attr[:-1]][
                    mesh.nn - 1 : mesh.nn + mesh.ns
                ]
                avg_s = np.sum(var_s) * mesh.dxs / param.ls
                self.__dict__[attr[:-1] + "s_avg"] = avg_s

                # Positive
                var_p = self.__dict__[attr[:-1]][mesh.nn + mesh.ns :]
                avg_p = np.sum(var_p) * mesh.dxp / param.lp
                self.__dict__[attr[:-1] + "p_avg"] = avg_p
