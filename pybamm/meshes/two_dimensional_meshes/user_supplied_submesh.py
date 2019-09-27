#
# User supplied two-dimensional scikit-fem submesh
#
import pybamm
from .base_scikit_fem_submesh import ScikitSubMesh2D


class GetUserSupplied2DSubMesh:
    """
    A class to generate a tensor product submesh on a 2D domain by using two user
    supplied vectors of edges: one for the y-direction and one for the z-direction.
    ----------
    edges : array_like
        The array of points which correspond to the edges of the mesh.
    """

    def __init__(self, nodes):
        self.nodes = nodes

    def __call__(self, lims, npts, tabs=None):
        return UserSupplied2DSubMesh(lims, npts, tabs, self.nodes)


class UserSupplied2DSubMesh(ScikitSubMesh2D):
    """
    A class to generate a tensor product submesh on a 2D domain by using two user
    supplied vectors of edges: one for the y-direction and one for the z-direction.
    Note: this mesh should be created using :class:`GetUserSupplied2DSubMesh`.
    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    tabs : dict
        A dictionary that contains information about the size and location of
        the tabs
    edges : array_like
        The array of points which correspond to the edges of the mesh.
    """

    def __init__(self, lims, npts, tabs, edges):

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

        # check that npts equals number of user-supplied edges
        if npts != len(edges):
            raise pybamm.GeometryError(
                """User-suppled edges has should have length npts but has length {}.
                 Number of points (npts) for domain {} is {}.""".format(
                    len(edges), spatial_var.domain, npts
                )
            )

        # check end points of edges agree with spatial_lims
        if edges[0] != spatial_lims["min"]:
            raise pybamm.GeometryError(
                """First entry of edges is {}, but should be equal to {}
                 for domain {}.""".format(
                    edges[0], spatial_lims["min"], spatial_var.domain
                )
            )
        if edges[-1] != spatial_lims["max"]:
            raise pybamm.GeometryError(
                """Last entry of edges is {}, but should be equal to {}
                for domain {}.""".format(
                    edges[-1], spatial_lims["max"], spatial_var.domain
                )
            )

        coord_sys = spatial_var.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)
