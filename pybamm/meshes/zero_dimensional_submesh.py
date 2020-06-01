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
    position : dict
        A dictionary that contains the position of the 0D submesh (a signle point)
        in space
    npts : dict, optional
        Number of points to be used. Included for compatibility with other meshes,
        but ignored by this mesh class

    **Extends:"": :class:`pybamm.SubMesh`
    """

    def __init__(self, position, npts=None):
        # Remove tabs
        position.pop("tabs", None)

        # check that only one variable passed in
        if len(position) != 1:
            raise pybamm.GeometryError("position should only contain a single variable")

        spatial_position = list(position.values())[0]
        self.nodes = np.array([spatial_position])
        self.edges = np.array([spatial_position])
        self.coord_sys = None
        self.npts = 1

    def add_ghost_meshes(self):
        # No ghost meshes to be added to this class
        pass
