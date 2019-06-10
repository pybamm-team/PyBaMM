#
# fenics meshes for use in PyBaMM
#
import pybamm

import numpy as np
import importlib

dolfin_spec = importlib.util.find_spec("dolfin")
if dolfin_spec is not None:
    dolfin = importlib.util.module_from_spec(dolfin_spec)
    dolfin_spec.loader.exec_module(dolfin)


class FenicsMesh2D:
    """ Submesh class.
        Contains the position of the nodes and the number of mesh points.

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
        if dolfin_spec is None:
            raise ImportError("dolfin is not installed")

        # get spatial variables
        spatial_vars = list(lims.keys())

        # set limits and number of points
        for var in spatial_vars:
            if var.name == "y":
                self.y_lims = lims[var]
                self.Ny = npts[var.id] - 1
            elif var.name == "z":
                self.z_lims = lims[var]
                self.Nz = npts[var.id] - 1
            else:
                raise pybamm.DomainError(
                    "spatial variable must be y or z not {}".format(var.name)
                )

        # check coordinate system agrees
        if spatial_vars[0].coord_sys == spatial_vars[1].coord_sys:
            self.coord_sys = spatial_vars[0].coord_sys
        else:
            raise pybamm.DomainError(
                """spatial variables should have the same domain,
                but have domians {} and {}""".format(
                    spatial_vars[0].coord_sys, spatial_vars[1].coord_sys
                )
            )

        # create mesh
        self.fem_mesh = dolfin.RectangleMesh(
            dolfin.Point(self.y_lims["min"], self.z_lims["min"]),
            dolfin.Point(self.y_lims["max"], self.z_lims["max"]),
            self.Ny,
            self.Nz,
        )

        # create SubDomain classes for the tabs
        self.negativetab = Tab(self.y_lims, self.z_lims, tabs["negative"])
        self.positivetab = Tab(self.y_lims, self.z_lims, tabs["positive"])

        # initialize mesh function for boundary domains
        boundary_markers = dolfin.MeshFunction(
            "size_t", self.fem_mesh, self.fem_mesh.topology().dim() - 1
        )
        boundary_markers.set_all(0)
        self.negativetab.mark(boundary_markers, 1)
        self.positivetab.mark(boundary_markers, 2)

        # create measure of parts of the boundary
        self.ds = dolfin.Measure(
            "ds", domain=self.fem_mesh, subdomain_data=boundary_markers
        )

        # create measure for domain
        self.dx = dolfin.dx

        # create function space (at the moment a lot of the code relies on using
        # degree 1 Lagrange elements, so this is hard coded in here to avoid users
        # changing the function space)
        self.degree = 1
        self.FunctionSpace = dolfin.FunctionSpace(
            self.fem_mesh, "Lagrange", self.degree
        )

        self.TrialFunction = dolfin.TrialFunction(self.FunctionSpace)
        self.TestFunction = dolfin.TestFunction(self.FunctionSpace)
        self.Function = dolfin.Function(self.FunctionSpace)

        # NOTE: below only works for degree 1 Lagrange elements
        # get number of mesh points (= to numbers of degrees of freedom)
        self.npts = np.size(self.Function.vector()[:])
        # get mesh coordinates in the same order as the degrees of freedom
        self.coordinates = self.fem_mesh.coordinates()[
            dolfin.dof_to_vertex_map(self.FunctionSpace)
        ]


# Create instance of dolfin.Subdomain class for the tabs
if dolfin_spec is not None:

    class Tab(dolfin.SubDomain):
        """
        A class to generate the subdomains for the tabs.
        """

        def __init__(self, y_lims, z_lims, tab):
            self.l_y = y_lims["max"]
            self.l_z = z_lims["max"]
            self.tab_location = [tab["y_centre"], tab["z_centre"]]
            self.tab_width = tab["width"]

            super().__init__()

        def inside(self, x, on_boundary):
            if dolfin.near(self.tab_location[1], self.l_z):
                # tab on top
                return dolfin.near(x[1], self.l_z) and dolfin.between(
                    x[0],
                    (
                        self.tab_location[0] - self.tab_width / 2,
                        self.tab_location[0] + self.tab_width / 2,
                    ),
                )
            elif dolfin.near(self.tab_location[1], 0.0):
                # tab on bottom
                return dolfin.near(x[1], 0.0) and dolfin.between(
                    x[0],
                    (
                        self.tab_location[0] - self.tab_width / 2,
                        self.tab_location[0] + self.tab_width / 2,
                    ),
                )
            elif dolfin.near(self.tab_location[0], 0.0):
                # tab on left
                return dolfin.near(x[0], 0.0) and dolfin.between(
                    x[1],
                    (
                        self.tab_location[1] - self.tab_width / 2,
                        self.tab_location[1] + self.tab_width / 2,
                    ),
                )
            elif dolfin.near(self.tab_location[0], self.l_y):
                # tab on right
                return dolfin.near(x[0], self.l_y) and dolfin.between(
                    x[1],
                    (
                        self.tab_location[1] - self.tab_width / 2,
                        self.tab_location[1] + self.tab_width / 2,
                    ),
                )
            else:
                raise pybamm.ModelError("tab location not valid")
