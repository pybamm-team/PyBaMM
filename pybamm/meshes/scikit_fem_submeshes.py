#
# scikit-fem meshes for use in PyBaMM
#
import pybamm

import numpy as np
import importlib

skfem_spec = importlib.util.find_spec("skfem")
if skfem_spec is not None:
    skfem = importlib.util.module_from_spec(skfem_spec)
    skfem_spec.loader.exec_module(skfem)


class Scikit2DSubMesh:
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

    def __init__(self, lims, npts, tabs):
        if skfem_spec is None:
            raise ImportError("scikit-fem is not installed")

        # check that two variables have been passed in
        if len(lims) != 2:
            raise pybamm.GeometryError(
                "lims should contain exactly two variables, not {}".format(len(lims))
            )

        # get spatial variables
        spatial_vars = list(lims.keys())

        # check coordinate system agrees
        if spatial_vars[0].coord_sys == spatial_vars[1].coord_sys:
            self.coord_sys = spatial_vars[0].coord_sys
        else:
            raise pybamm.DomainError(
                """spatial variables should have the same coordinate system,
                but have coordinate systems {} and {}""".format(
                    spatial_vars[0].coord_sys, spatial_vars[1].coord_sys
                )
            )

        # set limits and number of points
        self.npts = 1
        self.edges = {}
        self.nodes = {}
        for var in spatial_vars:
            if var.name not in ["y", "z"]:
                raise pybamm.DomainError(
                    "spatial variable must be y or z not {}".format(var.name)
                )
            else:
                self.npts *= npts[var.id]
                self.edges[var.name] = np.linspace(
                    lims[var]["min"], lims[var]["max"], npts[var.id]
                )
                self.nodes[var.name] = (
                    self.edges[var.name][1:] + self.edges[var.name][:-1]
                ) / 2

        # create mesh
        self.fem_mesh = skfem.MeshTri.init_tensor(self.edges["y"], self.edges["z"])

        # get coordinates (returns a vector size 2*(Ny*Nz))
        self.coordinates = self.fem_mesh.p

        # create elements and basis
        self.element = skfem.ElementTriP1()
        self.basis = skfem.InteriorBasis(self.fem_mesh, self.element)
        self.facet_basis = skfem.FacetBasis(self.fem_mesh, self.element)

        # get degrees of freedom which correspond to tabs
        self.negative_tab = self.basis.get_dofs(
            lambda y, z: self.on_boundary(y, z, tabs["negative"])
        ).all()
        self.positive_tab = self.basis.get_dofs(
            lambda y, z: self.on_boundary(y, z, tabs["positive"])
        ).all()

    def on_boundary(self, y, z, tab):
        """
        A method to get the degrees of freedom corresponding to the subdomains
        for the tabs.
        """

        l_y = self.edges["y"][-1]
        l_z = self.edges["z"][-1]

        def near(x, point, tol=1e-6):
            return abs(x - point) < tol

        def between(x, interval, tol=1e-6):
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
