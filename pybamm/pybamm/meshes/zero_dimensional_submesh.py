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
    """

    def __init__(self, position, npts=None):
        # Remove tabs
        position.pop("tabs", None)

        # check that only one variable passed in
        if len(position) != 1:
            raise pybamm.GeometryError("position should only contain a single variable")

        # extract the position
        position = next(iter(position.values()))
        spatial_position = position["position"]
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
