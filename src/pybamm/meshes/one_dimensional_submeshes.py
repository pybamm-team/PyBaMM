#
# One-dimensional submeshes
#
import pybamm
from .meshes import SubMesh

import numpy as np


class SubMesh1D(SubMesh):
    """
    1D submesh class.
    Contains the position of the nodes, the number of mesh points, and
    (optionally) information about the tab locations.

    Parameters
    ----------
    edges : array_like
        An array containing the points corresponding to the edges of the submesh
    coord_sys : string
        The coordinate system of the submesh
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, edges, coord_sys, tabs=None):
        self.edges = edges
        self.nodes = (self.edges[1:] + self.edges[:-1]) / 2
        self.d_edges = np.diff(self.edges)
        self.d_nodes = np.diff(self.nodes)
        self.npts = self.nodes.size
        self.coord_sys = coord_sys
        self.internal_boundaries = []

        # Add tab locations in terms of "left" and "right"
        if tabs and "negative tab" not in tabs.keys():
            self.tabs = {}
            l_z = self.edges[-1]

            def near(x, point, tol=3e-16):
                return abs(x - point) < tol

            for tab in ["negative", "positive"]:
                tab_location = tabs[tab]["z_centre"]
                if near(tab_location, 0):
                    self.tabs[tab + " tab"] = "left"
                elif near(tab_location, l_z):
                    self.tabs[tab + " tab"] = "right"
                else:
                    raise pybamm.GeometryError(
                        f"{tab} tab located at {tab_location}, "
                        f"but must be at either 0 or {l_z}"
                    )
        elif tabs:
            # tabs have already been calculated by a serialised model
            self.tabs = tabs

    def read_bounds(self, domain):
        # Separate limits and tabs
        bounds = domain.dimension_bounds
        if len(bounds) > 1:
            raise pybamm.GeometryError("Domain has more than one dimension")

        return bounds[0]

    def to_json(self):
        json_dict = {
            "edges": self.edges.tolist(),
            "coord_sys": self.coord_sys,
        }

        if hasattr(self, "tabs"):
            json_dict["tabs"] = self.tabs

        return json_dict


class Uniform1DSubMesh(SubMesh1D):
    """
    A class to generate a uniform submesh on a 1D domain

    Parameters
    ----------
    domain : :class:`pybamm.Domain`
        The domain to generate a submesh for
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, domain, npts, tabs=None):
        bounds = self.read_bounds(domain)
        edges = np.linspace(bounds[0], bounds[1], npts + 1)
        coord_sys = domain.coord_sys
        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)

    @classmethod
    def _from_json(cls, snippet: dict):
        instance = cls.__new__(cls)

        tabs = snippet["tabs"] if "tabs" in snippet.keys() else None

        super(Uniform1DSubMesh, instance).__init__(
            np.array(snippet["edges"]), snippet["coord_sys"], tabs=tabs
        )

        return instance


class Exponential1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain in which the points are clustered
    close to one or both of boundaries using an exponential formula on the interval
    [a,b].

    If side is "left", the gridpoints are given by

    .. math::
        x_{k} = (b-a) +
        \\frac{\\mathrm{e}^{\\alpha k / N} - 1}{\\mathrm{e}^{\\alpha} - 1} + a,

    for k = 1, ..., N, where N is the number of nodes.

    Is side is "right", the gridpoints are given by

    .. math::
        x_{k} = (b-a) +
        \\frac{\\mathrm{e}^{-\\alpha k / N} - 1}{\\mathrm{e}^{-\\alpha} - 1} + a,

    for k = 1, ..., N.

    If side is "symmetric", the first half of the interval is meshed using the
    gridpoints

    .. math::
        x_{k} = (b/2-a) +
        \\frac{\\mathrm{e}^{\\alpha k / N} - 1}{\\mathrm{e}^{\\alpha} - 1} + a,

    for k = 1, ..., N. The grid spacing is then reflected to contruct the grid
    on the full interval [a,b].

    In the above, alpha is a stretching factor. As the number of gridpoints tends
    to infinity, the ratio of the largest and smallest grid cells tends to exp(alpha).

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    side : str, optional
        Whether the points are clustered near to the left or right boundary,
        or both boundaries. Can be "left", "right" or "symmetric". Default is
        "symmetric"
    stretch : float, optional
        The factor (alpha) which appears in the exponential. If side is "symmetric"
        then the default stretch is 1.15. If side is "left" or "right" then the
        default stretch is 2.3.
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, domain, npts, side="symmetric", stretch=None, tabs=None):
        bounds = self.read_bounds(domain)
        a, b = bounds
        coord_sys = domain.coord_sys

        # Set stretch if not provided
        if not stretch:
            if side == "symmetric":
                stretch = 1.15
            elif side in ["left", "right"]:
                stretch = 2.3

        # Create edges accoriding to "side"
        if side == "left":
            ii = np.array(range(0, npts + 1))
            edges = (b - a) * (np.exp(stretch * ii / npts) - 1) / (
                np.exp(stretch) - 1
            ) + a

        elif side == "right":
            ii = np.array(range(0, npts + 1))
            edges = (b - a) * (np.exp(-stretch * ii / npts) - 1) / (
                np.exp(-stretch) - 1
            ) + a

        elif side == "symmetric":
            # Mesh half-interval [a, b/2]
            if npts % 2 == 0:
                ii = np.array(range(0, int((npts) / 2)))
            else:
                ii = np.array(range(0, int((npts + 1) / 2)))
            x_exp_left = (b / 2 - a) * (np.exp(stretch * ii / npts) - 1) / (
                np.exp(stretch) - 1
            ) + a

            # Refelct mesh
            x_exp_right = b * np.ones_like(x_exp_left) - (x_exp_left[::-1] - a)

            # Combine left and right halves of the mesh, adding a node at the
            # centre if npts is even (odd number of edges)
            if npts % 2 == 0:
                edges = np.concatenate((x_exp_left, [(a + b) / 2], x_exp_right))
            else:
                edges = np.concatenate((x_exp_left, x_exp_right))

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class Chebyshev1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain using Chebyshev nodes on the
    interval (a, b), given by

    .. math::
        x_{k} = \\frac{1}{2}(a+b) + \\frac{1}{2}(b-a) \\cos(\\frac{2k-1}{2N}\\pi),

    for k = 1, ..., N, where N is the number of nodes. Note: this mesh then
    appends the boundary edges, so that the mesh edges are given by

    .. math::
        a < x_{1} < ... < x_{N} < b.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, domain, npts, tabs=None):
        bounds = self.read_bounds(domain)

        # Create N Chebyshev nodes in the interval (a,b)
        N = npts - 1
        ii = np.array(range(1, N + 1))
        a, b = bounds
        x_cheb = (a + b) / 2 + (b - a) / 2 * np.cos((2 * ii - 1) * np.pi / 2 / N)

        # Append the boundary nodes. Note: we need to flip the order the Chebyshev
        # nodes as they are created in descending order.
        edges = np.concatenate(([a], np.flip(x_cheb), [b]))
        coord_sys = domain.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class UserSupplied1DSubMesh(SubMesh1D):
    """
    A class to generate a submesh on a 1D domain from a user supplied array of
    edges.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on each
        spatial variable. Note: the number of nodes (located at the cell centres)
        is npts, and the number of edges is npts+1.
    edges : array_like
        The array of points which correspond to the edges of the mesh.
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, domain, npts, edges=None, tabs=None):
        if edges is None:
            raise pybamm.GeometryError("User mesh requires parameter 'edges'")

        bounds = self.read_bounds(domain)

        if (npts + 1) != len(edges):
            raise pybamm.GeometryError(
                f"User-suppled edges has should have length (npts + 1) but has length "
                f"{len(edges)}. Number of points (npts) for domain is {npts}."
            )

        # check end points of edges agree with spatial_lims
        if edges[0] != bounds[0]:
            raise pybamm.GeometryError(
                f"First entry of edges is {edges[0]}, but should be equal to "
                f"{bounds[0]}."
            )
        if edges[-1] != bounds[1]:
            raise pybamm.GeometryError(
                f"Last entry of edges is {edges[-1]}, but should be equal to "
                f"{bounds[1]}."
            )

        coord_sys = domain.coord_sys

        super().__init__(edges, coord_sys=coord_sys, tabs=tabs)


class SpectralVolume1DSubMesh(SubMesh1D):
    """
    A class to subdivide any mesh to incorporate Chebyshev collocation
    Control Volumes. Note that the Spectral Volume method is optimized
    to only work with this submesh. The underlying theory could use any
    mesh with the right number of nodes, but in 1D the only sensible
    choice are the Chebyshev collocation points.

    Parameters
    ----------
    lims : dict
        A dictionary that contains the limits of the spatial variables
    npts : dict
        A dictionary that contains the number of points to be used on
        each spatial variable. Note: the number of nodes (located at the
        cell centres) is npts, and the number of edges is npts+1.
    order : int, optional
        The order of the Spectral Volume method that is to be used with
        this submesh. The default is 2, the same as the default for the
        SpectralVolume class. If the orders of the submesh and the
        Spectral Volume method don't match, the method will fail.
    tabs : dict, optional
        A dictionary that contains information about the size and location of
        the tabs
    """

    def __init__(self, domain, npts, edges=None, order=2, tabs=None):
        bounds = self.read_bounds(domain)

        # default: Spectral Volumes of equal size
        if edges is None:
            edges = np.linspace(bounds[0], bounds[1], npts + 1)
        # check that npts + 1 equals number of user-supplied edges
        elif (npts + 1) != len(edges):
            raise pybamm.GeometryError(
                "User-suppled edges should have length (npts + 1) but has len"
                f"gth {len(edges)}. Number of points (npts) is {npts}."
            )

        # check end points of edges agree with spatial_lims
        if edges[0] != bounds[0]:
            raise pybamm.GeometryError(
                f"First entry of edges is {edges[0]}, but should be equal to "
                f"{bounds[0]}."
            )
        if edges[-1] != bounds[1]:
            raise pybamm.GeometryError(
                f"Last entry of edges is {edges[-1]}, but should be equal to "
                f"{bounds[1]}."
            )

        coord_sys = domain.coord_sys

        array = np.array(
            [
                ((order + 1) - 1 - 2 * i) / (2 * (order + 1) - 2)
                for i in range(order + 1)
            ]
        )
        cv_edges = np.array(
            [edges[0]]
            + [
                x
                for (a, b) in zip(edges[:-1], edges[1:])
                for x in np.flip(a + 0.5 * (b - a) * (1 + np.sin(np.pi * array)))[1:]
            ]
        )

        self.sv_edges = edges
        self.sv_nodes = (edges[:-1] + edges[1:]) / 2
        self.d_sv_edges = np.diff(self.sv_edges)
        self.d_sv_nodes = np.diff(self.sv_nodes)
        self.order = 2
        # The Control Volume edges and nodes are assigned to the
        # "edges" and "nodes" properties. This makes some of the
        # code of FiniteVolume directly applicable.
        super().__init__(cv_edges, coord_sys=coord_sys, tabs=tabs)
