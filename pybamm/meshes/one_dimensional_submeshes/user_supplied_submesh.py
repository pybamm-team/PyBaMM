#
# User supplied one-dimensional submesh
#
import pybamm


class GetUserSupplied1DSubMesh:
    """
    A class to generate a submesh on a 1D domain using a user supplied vector of
    nodes.
    Parameters
    ----------
    edges : array_like
        The array of points which correspond to the edges of the mesh.
    """

    def __init__(self, nodes):
        self.nodes = nodes

    def __call__(self, lims, npts, tabs=None):
        return UserSupplied1DSubMesh(lims, npts, tabs, self.nodes)


class UserSupplied1DSubMesh(pybamm.SubMesh1D):
    """
    A class to generate a submesh on a 1D domain from a user supplied array of
    nodes.
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

        # check that only one variable passed in
        if len(lims) != 1:
            raise pybamm.GeometryError("lims should only contain a single variable")

        spatial_var = list(lims.keys())[0]
        spatial_lims = lims[spatial_var]
        npts = npts[spatial_var.id]

        # check that npts + 1 equals number of user-supplied edges
        if (npts + 1) != len(edges):
            raise pybamm.GeometryError(
                """User-suppled edges has should have length (npts + 1) but has length {}.
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
