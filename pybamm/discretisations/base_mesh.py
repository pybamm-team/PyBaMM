#
# Mesh class for space and time discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np

KNOWN_DOMAINS = [
    "negative electrode",
    "separator",
    "positive electrode",
    "whole cell",
    "test",
]


class BaseMesh(dict):
    """A base mesh class.
    Mesh inherits from dict, with entries in the dictionary being submeshes (time and
    space)

    Parameters
    ----------
    param : :class:`pybamm.parameters.ParameterValues` instance
        The parameters defining the subdomain sizes.
    target_npts : int
        The target number of points in each domain. The mesh will be created
        in such a way that the cell sizes are as similar as possible between
        domains.
    tsteps : int
        The number of time steps to take
    tend : float
        The finishing time for the simulation

    """

    def __init__(self, param, target_npts=10, tsteps=100, tend=1):

        # Time
        self["time"] = np.linspace(0, tend, tsteps)

        # submesh class
        self.submeshclass = BaseSubmesh


class BaseSubmesh(object):
    """Base submesh class.
    Contains the position of the nodes and the number of mesh points.

    Parameters
    ----------
    nodes : array type
        The positions of the nodes of the submesh
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.npts = nodes.size
