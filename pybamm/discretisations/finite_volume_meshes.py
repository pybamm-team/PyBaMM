#
# Mesh class for space and time discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class FiniteVolumeMacroMesh(pybamm.BaseMesh):
    """A Finite Volumes mesh for the 1D macroscale.

    **Extends**: :class:`BaseMesh`

    """

    def __init__(self, param, target_npts=10, tsteps=100, tend=1):

        super().__init__(param, target_npts, tsteps, tend)

        # submesh class
        self.submeshclass = FiniteVolumeSubmesh

        # Space (macro)
        Ln, Ls, Lp = param["Ln"], param["Ls"], param["Lp"]
        L = Ln + Ls + Lp
        ln, ls, lp = Ln / L, Ls / L, Lp / L
        # We aim to create the grid as uniformly as possible
        targetmeshsize = min(ln, ls, lp) / target_npts

        # Negative electrode
        self.neg_mesh_points = round(ln / targetmeshsize) + 1
        self["negative electrode"] = self.submeshclass(
            np.linspace(0.0, ln, self.neg_mesh_points)
        )

        # Separator
        self.sep_mesh_points = round(ls / targetmeshsize) + 1
        self["separator"] = self.submeshclass(
            np.linspace(ln, ln + ls, self.sep_mesh_points)
        )

        # Positive electrode
        self.pos_mesh_points = round(lp / targetmeshsize) + 1
        self["positive electrode"] = self.submeshclass(
            np.linspace(ln + ls, 1.0, self.pos_mesh_points)
        )

        # Whole cell
        self.total_mesh_points = (
            self.neg_mesh_points + (self.sep_mesh_points - 2) + self.pos_mesh_points
        )
        self["whole cell"] = self.combine_submeshes(
            "negative electrode", "separator", "positive electrode"
        )

        # Add ghost meshes for ghost cells for Dirichlet boundary conditions
        self.add_ghost_meshes()

    def combine_submeshes(self, *submeshnames):
        """Combine submeshes into a new submesh, using self.submeshclass
        Raises pybamm.DomainError if submeshes to be combined do not match up (edges are
        not aligned).

        Parameters
        ----------
        submeshnames: list of str
            The names of the submeshes to be combined

        Returns
        -------
        submesh: :class:`self.submeshclass`
            A new submesh with the class defined by self.submeshclass
        """
        # Check that the final edge of each submesh is the same as the first edge of the
        # next submesh
        edges_aligned = all(
            [
                self[submeshnames[i]].edges[-1] == self[submeshnames[i + 1]].edges[0]
                for i in range(len(submeshnames) - 1)
            ]
        )
        if edges_aligned:
            # Combine submeshes, being careful not to double-count repeated edges at the
            # intersection of submeshes
            combined_submesh_edges = np.concatenate(
                [self[submeshnames[0]].edges]
                + [self[submeshname].edges[1:] for submeshname in submeshnames[1:]]
            )
            return self.submeshclass(combined_submesh_edges)
        else:
            raise pybamm.DomainError("submesh edges are not aligned")

    def add_ghost_meshes(self):
        """
        Create meshes for potential ghost nodes on either side of each submesh, using
        self.submeshclass
        This will be useful for calculating the gradient with Dirichlet BCs.
        """
        # Get all submeshes relating to space (i.e. exclude time)
        submeshes = [
            (name, submesh) for name, submesh in self.items() if name != "time"
        ]
        for submeshname, submesh in submeshes:
            edges = submesh.edges
            # left ghost cell: two edges, one node, to the left of existing submesh
            lgs_edges = np.array([2 * edges[0] - edges[1], edges[0]])
            self[submeshname + "_left ghost cell"] = self.submeshclass(lgs_edges)
            # right ghost cell: two edges, one node, to the right of existing submesh
            rgs_edges = np.array([edges[-1], 2 * edges[-1] - edges[-2]])
            self[submeshname + "_right ghost cell"] = self.submeshclass(rgs_edges)


class FiniteVolumeSubmesh:
    """A submesh for finite volumes.

    The mesh is defined by its edges; then node positions, diffs and mesh size are
    calculated from the edge positions.

    Parameters
    ----------
    edges : :class:`numpy.array`
        The position of the edges of the cells

    """

    def __init__(self, edges):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size
