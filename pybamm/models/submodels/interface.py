#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Interface(object):
    """
    Equations for the electrode-electrolyte interface in a single electrode.

    Parameters
    ----------
    param : dict
        The parameters of the simulation for reactions in the electrode.
    mesh : dict
        The mesh in the electrode.
    """

    def __init__(self, param, mesh):
        self.param = param
        self.mesh = mesh

    def reaction(self, vars):
        """
        Calculates the interfacial current density in an electrode.

        Parameters
        ----------
        vars : dict
            The variables in the electrode

        Returns
        -------
        j : array_like
            The interfacial current density in the electrode.

        """
        raise NotImplementedError


class ButlerVolmer(Interface):
    """
    Butler-Volmer kinetics.
    """

    def __init__(self, param, mesh):
        super().__init__(param, mesh)

    def reaction(self, vars):
        """
        See :meth:`Interface.reaction`
        """
        c, e = vars["c"], vars["e"]

        assert c.shape == e.shape
        assert e.shape[0] == self.mesh.npts
        return self.param.j0(c) * np.sinh(e - self.param.U(c))


class HomogeneousReaction(Interface):
    """
    Spatially homogeneous reaction in each electrode.
    """

    def __init__(self, param, mesh):
        super().__init__(param, mesh)

    def reaction(self, vars):
        """
        See :meth:`Interface.reaction`
        """
        j = self.param.j_avg(vars.t) * np.ones_like(self.mesh.x.centres)

        return j
