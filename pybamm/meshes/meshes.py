#
# Native PyBaMM Meshes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np
import pybamm

KNOWN_DOMAINS = [
    "negative electrode",
    "separator",
    "positive electrode",
    "test",
    "negative particle",
    "positive particle",
]


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
        self.submesh_pts = submesh_pts
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
            return pybamm.SubMesh1D(combined_submesh_edges)
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
            self[submeshname + "_left ghost cell"] = pybamm.SubMesh1D(lgs_edges)

            # right ghost cell: two edges, one node, to the right of existing submesh
            rgs_edges = np.array([edges[-1], 2 * edges[-1] - edges[-2]])
            self[submeshname + "_right ghost cell"] = pybamm.SubMesh1D(rgs_edges)
