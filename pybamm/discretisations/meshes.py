#
# Native PyBaMM Meshes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
import pybamm

KNOWN_DOMAINS = ["negative electrode", "separator", "positive electrode", "test"]


class Mesh(dict):
    """
    Mesh contains the submeshes

    **Extends**: dict

    Parameters
    ----------

    geometry : :class: `Geometry`
        contains the geometry of the problem
    submesh_types: dict
        contains the types of submeshes to use (e.g. Uniform1DSubMesh)
    submesh_pts: dict
        contains the number of points on each subdomain

    """

    def __init__(self, geometry, submesh_types, submesh_pts):
        super().__init__()
        for domain in geometry:
            submesh_type = submesh_types[domain]
            submesh_pt = submesh_pts[domain]
            self[domain] = submesh_type(geometry[domain], submesh_pt)

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
            return SubMesh1D(combined_submesh_edges)
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
            self[submeshname + "_left ghost cell"] = SubMesh1D(lgs_edges)

            # right ghost cell: two edges, one node, to the right of existing submesh
            rgs_edges = np.array([edges[-1], 2 * edges[-1] - edges[-2]])
            self[submeshname + "_right ghost cell"] = SubMesh1D(rgs_edges)


class SubMesh1D:
    """ Submesh class.
        Contains the position of the nodes and the number of mesh points.

        Parameters
        ----------
        domain : dict
            A dictionary that contains the limits of the spatial variables
        npts : dict
            A dictionary that contains the number of points to be used on each
            spatial variable
        """

    def __init__(self, edges):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size


class Uniform1DSubMesh(SubMesh1D):
    """
    A class to generate a uniform submesh on a 1D domain
    """

    def __init__(self, lims, npts):

        # currently accept lims and npts as dicts. This may get changed at a future
        # date depending on the form of mesh we desire for 2D/3D

        spatial_lims = list(lims.values())[0]
        npts = list(npts.values())[0]

        edges = np.linspace(spatial_lims["min"], spatial_lims["max"], npts + 1)

        super().__init__(edges)
