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

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh

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
        for var, mesh in self.model.pde_variables:
            end = start + mesh.npts
            self.__dict__[var] = y[start:end]
            start = end
            # Split 'x' variables further into n, s and p
            if mesh == self.mesh.x:
                (
                    self.__dict__[var + "n"],
                    self.__dict__[var + "s"],
                    self.__dict__[var + "p"],
                ) = np.split(
                    self.__dict__[var],
                    np.cumsum([self.mesh.xn.npts, self.mesh.xs.npts]),
                )

    @property
    def neg(self):
        """Variables in the negative electrode."""
        neg_vars = {}
        for attr, value in self.__dict__.items():
            if attr[-1] == "n":
                neg_vars[attr[:-1]] = value
        return neg_vars

    @property
    def pos(self):
        """Variables for the positive electrode."""
        pos_vars = {}
        for attr, value in self.__dict__.items():
            if attr[-1] == "p":
                pos_vars[attr[:-1]] = value
        return pos_vars

    def set_reaction_vars(self, reaction_vars):
        for name, value in reaction_vars.items():
            self.__dict__[name] = value
            (
                self.__dict__[name + "n"],
                self.__dict__[name + "s"],
                self.__dict__[name + "p"],
            ) = np.split(
                self.__dict__[name], np.cumsum([self.mesh.nn - 1, self.mesh.ns + 1])
            )

    def average(self):
        """Average variable attributes over the relevant (sub)domain."""
        param = self.model.param.geometric
        mesh = self.mesh
        old_keys = list(self.__dict__.keys())
        for attr in old_keys:
            if attr in ["c"]:
                avg = np.dot(self.__dict__[attr].T, mesh.x.d_edges)
                self.__dict__[attr + "_avg"] = avg

            elif attr[-1] == "n":
                # Negative
                avg = np.dot(self.__dict__[attr].T, mesh.xn.d_edges) / param.ln
                self.__dict__[attr + "_avg"] = avg

            elif attr[-1] == "s":
                # Separator
                avg = np.dot(self.__dict__[attr].T, mesh.xs.d_edges) / param.ls
                self.__dict__[attr + "_avg"] = avg

            elif attr[-1] == "p":
                # Positive
                avg = np.dot(self.__dict__[attr].T, mesh.xp.d_edges) / param.lp
                self.__dict__[attr + "_avg"] = avg
