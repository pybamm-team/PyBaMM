#
# Uniform scikit-fem mesh for use in PyBaMM
#
import pybamm
from .base_scikit_fem_submesh import ScikitSubMesh2D

import numpy as np


class ScikitUniform2DSubMesh(ScikitSubMesh2D):
    """
    Contains information about the 2D finite element mesh with uniform grid
    spacing (can be different spacing in y and z).
    Note: This class only allows for the use of piecewise-linear triangular
    finite elements.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of each
        spatial variable
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable
    tabs : dict
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, lims, npts, tabs):

        # check that two variables have been passed in
        if len(lims) != 2:
            raise pybamm.GeometryError(
                "lims should contain exactly two variables, not {}".format(len(lims))
            )

        # get spatial variables
        spatial_vars = list(lims.keys())

        # check coordinate system agrees
        if spatial_vars[0].coord_sys == spatial_vars[1].coord_sys:
            coord_sys = spatial_vars[0].coord_sys
        else:
            raise pybamm.DomainError(
                """spatial variables should have the same coordinate system,
                but have coordinate systems {} and {}""".format(
                    spatial_vars[0].coord_sys, spatial_vars[1].coord_sys
                )
            )

        # compute edges
        edges = {}
        for var in spatial_vars:
            if var.name not in ["y", "z"]:
                raise pybamm.DomainError(
                    "spatial variable must be y or z not {}".format(var.name)
                )
            else:
                edges[var.name] = np.linspace(
                    lims[var]["min"], lims[var]["max"], npts[var.id]
                )

        super().__init__(edges, coord_sys, tabs)
