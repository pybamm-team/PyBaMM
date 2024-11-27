#
# Zero dimensional submesh
#
import pybamm
from .meshes import SubMesh

import numpy as np


class SubMesh0D(SubMesh):
    """
    0D submesh class.
    Contains the position of the node.

    Parameters
    ----------
    domain : :class:`pybamm.Domain`
        The domain to generate a submesh for
    npts : dict, optional
        Number of points to be used. Included for compatibility with other meshes,
        but ignored by this mesh class
    """

    def __init__(self, domain, npts=None):
        # check that only one variable passed in
        if len(domain.dimensions) != 1:
            raise pybamm.GeometryError("position should only contain a single variable")

        # check that the bounds are equal
        bounds = domain.dimension_bounds[0]
        if bounds[0] != bounds[1]:
            raise pybamm.GeometryError("Bounds are not equal")

        # extract the position
        spatial_position = bounds[0]
        self.nodes = np.array([spatial_position])
        self.edges = np.array([spatial_position])
        self.coord_sys = None
        self.npts = 1

    @classmethod
    def _from_json(cls, snippet):
        instance = cls.__new__(cls)

        instance.nodes = np.array(snippet["spatial_position"])
        instance.edges = np.array(snippet["spatial_position"])
        instance.coord_sys = None
        instance.npts = 1

        return instance

    def add_ghost_meshes(self):
        # No ghost meshes to be added to this class
        pass

    def to_json(self):
        json_dict = {
            "spatial_position": self.nodes.tolist(),
        }
        return json_dict
