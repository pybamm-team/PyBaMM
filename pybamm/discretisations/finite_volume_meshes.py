#
# Mesh class for space and time discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class FiniteVolumeMacroMesh(pybamm.BaseMesh):
    """A Finite Volumes mesh for the 1D macroscale.

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

        super().__init__(param, target_npts, tsteps, tend)

        # Space (macro)
        ln, ls, lp = param.geometric.ln, param.geometric.ls, param.geometric.lp
        # We aim to create the grid as uniformly as possible
        targetmeshsize = min(ln, ls, lp) / target_npts

        # Negative electrode
        self.neg_mesh_points = round(ln / targetmeshsize) + 1
        self.set_submesh(
            "negative_electrode", np.linspace(0.0, ln, self.neg_mesh_points)
        )

        # Separator
        self.sep_mesh_points = round(ls / targetmeshsize) + 1
        self.set_submesh("separator", np.linspace(ln, ln + ls, self.sep_mesh_points))

        # Positive electrode
        self.pos_mesh_points = round(lp / targetmeshsize) + 1
        self.set_submesh(
            "positive_electrode", np.linspace(ln + ls, 1.0, self.pos_mesh_points)
        )

        # Whole cell
        self.total_mesh_points = (
            self.neg_mesh_points + (self.sep_mesh_points - 2) + self.pos_mesh_points
        )
        self.set_submesh(
            "whole_cell",
            np.concatenate(
                [
                    self.negative_electrode.edges,
                    self.separator.edges[1:-1],
                    self.positive_electrode.edges,
                ]
            ),
        )

    def set_submesh(self, submesh, edges):
        setattr(self, "_" + submesh, _SubMesh(edges))


class _SubMesh:
    def __init__(self, edges):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size
