#
# Base scikit-fem mesh for use in PyBaMM
#
import pybamm

import skfem


class ScikitSubMesh2D:
    """ Submesh class.
        Contains information about the 2D finite element mesh.
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
    def __init__(self, edges, coord_sys, tabs):
        self.edges = edges
        self.nodes = dict.fromkeys(["y", "z"])
        for var in self.nodes.keys():
            self.nodes[var] = (self.edges[var][1:] + self.edges[var][:-1]) / 2
        self.npts = len(self.edges["y"]) * len(self.edges["z"])
        self.coord_sys = coord_sys

        # create mesh
        self.fem_mesh = skfem.MeshTri.init_tensor(self.edges["y"], self.edges["z"])

        # get coordinates (returns a vector size 2*(Ny*Nz))
        self.coordinates = self.fem_mesh.p

        # create elements and basis
        self.element = skfem.ElementTriP1()
        self.basis = skfem.InteriorBasis(self.fem_mesh, self.element)
        self.facet_basis = skfem.FacetBasis(self.fem_mesh, self.element)

        # get degrees of freedom and facets which correspond to tabs, and
        # create facet basis for sub regions
        self.negative_tab_dofs = self.basis.get_dofs(
            lambda x: self.on_boundary(x[0], x[1], tabs["negative"])
        ).all()
        self.positive_tab_dofs = self.basis.get_dofs(
            lambda x: self.on_boundary(x[0], x[1], tabs["positive"])
        ).all()
        self.negative_tab_facets = self.fem_mesh.facets_satisfying(
            lambda x: self.on_boundary(x[0], x[1], tabs["negative"])
        )
        self.positive_tab_facets = self.fem_mesh.facets_satisfying(
            lambda x: self.on_boundary(x[0], x[1], tabs["positive"])
        )
        self.negative_tab_basis = skfem.FacetBasis(
            self.fem_mesh, self.element, facets=self.negative_tab_facets
        )
        self.positive_tab_basis = skfem.FacetBasis(
            self.fem_mesh, self.element, facets=self.positive_tab_facets
        )

    def on_boundary(self, y, z, tab):
        """
        A method to get the degrees of freedom corresponding to the subdomains
        for the tabs.
        """

        l_y = self.edges["y"][-1]
        l_z = self.edges["z"][-1]

        def near(x, point, tol=3e-16):
            return abs(x - point) < tol

        def between(x, interval, tol=3e-16):
            return x > interval[0] - tol and x < interval[1] + tol

        # Tab on top
        if near(tab["z_centre"], l_z):
            tab_left = tab["y_centre"] - tab["width"] / 2
            tab_right = tab["y_centre"] + tab["width"] / 2
            return [
                near(Z, l_z) and between(Y, [tab_left, tab_right]) for Y, Z in zip(y, z)
            ]
        # Tab on bottom
        elif near(tab["z_centre"], 0):
            tab_left = tab["y_centre"] - tab["width"] / 2
            tab_right = tab["y_centre"] + tab["width"] / 2
            return [
                near(Z, 0) and between(Y, [tab_left, tab_right]) for Y, Z in zip(y, z)
            ]
        # Tab on left
        elif near(tab["y_centre"], 0):
            tab_bottom = tab["z_centre"] - tab["width"] / 2
            tab_top = tab["z_centre"] + tab["width"] / 2
            return [
                near(Y, 0) and between(Z, [tab_bottom, tab_top]) for Y, Z in zip(y, z)
            ]
        # Tab on right
        elif near(tab["y_centre"], l_y):
            tab_bottom = tab["z_centre"] - tab["width"] / 2
            tab_top = tab["z_centre"] + tab["width"] / 2
            return [
                near(Y, l_y) and between(Z, [tab_bottom, tab_top]) for Y, Z in zip(y, z)
            ]
        else:
            raise pybamm.GeometryError("tab location not valid")
