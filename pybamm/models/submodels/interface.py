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
    subparam : :class:`pybamm.BaseParameterValues.Parameters` subclass instance
        The parameters of the simulation for reactions in the electrode.
    submesh : :class:`pybamm.mesh.Mesh` subclass instance
        The mesh in the electrode.
    """

    def __init__(self, subparam, submesh):
        self.subparam = subparam
        self.submesh = submesh

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

    def __init__(self, subparam, submesh):
        super().__init__(subparam, submesh)

    def reaction(self, vars):
        """
        See :meth:`Interface.reaction`
        """
        c, e = vars.c, vars.e

        assert c.shape == e.shape
        assert e.shape[0] == self.submesh.npts
        return self.subparam.j0(c) * np.sinh(e - self.subparam.U(c))


class HomogeneousReaction(Interface):
    """
    Spatially homogeneous reaction in each electrode.
    """

    def __init__(self, subparam, submesh):
        super().__init__(subparam, submesh)

    def reaction(self, vars):
        """
        See :meth:`Interface.reaction`
        """
        j = self.subparam.j_avg(vars.t) * np.ones_like(self.submesh.nodes)

        return j
