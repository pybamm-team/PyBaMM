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
    y_edges : array_like
        The array of points which correspond to the edges in the y direction
        of the mesh.
    z_edges : array_like
        The array of points which correspond to the edges in the z direction
        of the mesh.
    """

    def __init__(self, y_edges, z_edges):
        self.y_edges = y_edges
        self.z_edges = z_edges

    def __call__(self, lims, npts, tabs=None):
        return UserSupplied2DSubMesh(lims, npts, tabs, self.y_edges, self.z_edges)


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
    y_edges : array_like
        The array of points which correspond to the edges in the y direction
        of the mesh.
    z_edges : array_like
        The array of points which correspond to the edges in the z direction
        of the mesh.
    """

    def __init__(self, lims, npts, tabs, y_edges, z_edges):

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

        # check and store edges
        edges = {"y": y_edges, "z": z_edges}
        for var in spatial_vars:

            # check that npts equals number of user-supplied edges
            if npts[var.id] != len(edges[var.name]):
                raise pybamm.GeometryError(
                    """User-suppled edges has should have length npts but has length {}.
                     Number of points (npts) for variable {} in
                     domain {} is {}.""".format(
                        len(edges[var.name]), var.name, var.domain, npts[var.id]
                    )
                )

            # check end points of edges agree with spatial_lims
            if edges[var.name][0] != lims[var]["min"]:
                raise pybamm.GeometryError(
                    """First entry of edges is {}, but should be equal to {}
                     for variable {} in domain {}.""".format(
                        edges[var.name][0], lims[var]["min"], var.name, var.domain
                    )
                )
            if edges[var.name][-1] != lims[var]["max"]:
                raise pybamm.GeometryError(
                    """Last entry of edges is {}, but should be equal to {}
                    for variable {} in domain {}.""".format(
                        edges[var.name][-1], lims[var]["max"], var.name, var.domain
                    )
                )

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)
