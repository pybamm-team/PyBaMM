#
# Zero dimensional submesh
#
import pybamm
from .meshes import MeshGenerator

import numpy as np


class SubMesh0D:
    """
    0D submesh class.
    Contains the position of the node.

    Parameters
    ----------
    position : dict
        A dictionary that contains the position of the spatial variable
    npts : dict, optional
        Number of points to be used. Included for compatibility with other meshes, but
        ignored by this mesh class
    """

    def __init__(self, position, npts=None):
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


class MeshGenerator0D(MeshGenerator):
    """
    A class to generate a submesh on a 1D domain.

    Parameters
    ----------

    submesh_type: str, optional
        The type of submeshes to use. Can be "Position". Default is "Position".
    submesh_params: dict, optional
        Contains any parameters required by the submesh.

    **Extends**: :class:`pybamm.MeshGenerator`
    """

    def __init__(self, submesh_type="Position", submesh_params=None):
        self.submesh_type = submesh_type
        self.submesh_params = submesh_params or {}

    def __call__(self, position, npts=None):

        if self.submesh_type == "Position":
            return SubMesh0D(position, npts)
        else:
            raise pybamm.GeometryError(
                "Submesh {} not recognised.".format(self.submesh_type)
            )
