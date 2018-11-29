#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Interface(object):
    """
    Equations for the electrode-electrolyte interface.

    Parameters
    ----------
    param : :class:`pybamm.parameters.Parameters` instance
        The parameters of the simulation
    mesh : :class:`pybamm.mesh.Mesh` instance
        The spatial and temporal discretisation.
    """

    def __init__(self, param, mesh):
        self.param = param
        self.mesh = mesh

    def butler_volmer(self, vars):
        """
        Calculates the interfacial current densities using Butler-Volmer kinetics.

        Parameters
        ----------
        vars : :class:`pybamm.variables.Variables` instance
            The variables of the simulation

        Returns
        -------
        j : array_like
            The interfacial current density.

        """
        assert vars.c.shape == vars.e.shape
        assert vars.e.shape[0] == len(self.mesh["xc"])
        return self.param["j0"](vars.c) * np.sinh(vars.e - self.param["U"](vars.c))

    def uniform_current_density(self, vars):
        """Calculates the interfacial current densities
        using Butler-Volmer kinetics.

        Parameters
        ----------
        vars : :class:`pybamm.variables.Variables` instance
            The variables of the simulation

        Returns
        -------
        j : array_like
            The interfacial current density.

        """
        mesh = self.mesh

        if domain == "xcn":
            j = self.param.icell(vars.t) / self.param.ln * np.ones_like(mesh.xcn)
        elif domain == "xcs":
            j = np.zeros_like(mesh.xcs)
        elif domain == "xcp":
            j = -self.param.icell(vars.t) / self.param.lp * np.ones_like(mesh.xcp)
        elif domain == "xc":
            j = np.concatenate(
                [
                    self.uniform_current_density(domain, t)
                    for domain in ["xcn", "xcs", "xcp"]
                ]
            )

        return j
