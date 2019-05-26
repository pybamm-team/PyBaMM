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
        npts : dict
            A dictionary that contains the limits of each
            spatial variable
        npts : dict
            A dictionary that contains the number of points to be used on each
            spatial variable
        tabs : dict
            A dictionary that contains information about the size and location of
            the tabs
        """

    def __init__(self, lims, npts, tabs, degree=1):

        # get limits
        # NOTE: This is currently only implemented for use in the specifc
        # case of the 2D current collector problem. Could be made more general.
        y_lims = lims["y"]
        z_lims = lims["z"]

        # get spatial variables
        # TO DO: check does dict get reodered?
        spatial_vars = list(lims.keys())
        self.Ny = npts[spatial_vars[0].id]
        self.Nz = npts[spatial_vars[1].id]

        self.degree = degree

        # check coordinate system agrees
        if spatial_vars[0].coord_sys == spatial_vars[1].coord_sys:
            self.coord_sys = spatial_vars[0].coord_sys
        else:
            raise pybamm.DomainError(
                "spatial variables should have the same domain, but have domians {} and {}".format(
                    spatial_vars[0].coord_sys, spatial_vars[1].coord_sys
                )
            )

        # create mesh and function space
        self.fem_mesh = dolfin.RectangleMesh(
            dolfin.Point(y_lims[0], z_lims[0]),
            dolfin.Point(y_lims[1], z_lims[1]),
            self.Ny,
            self.Nz,
        )
        self.FunctionSpace = dolfin.FunctionSpace(
            self.fem_mesh, "Lagrange", self.degree
        )

        self.TrialFunction = dolfin.TrialFunction(self.FunctionSpace)
        self.TestFunction = dolfin.TestFunction(self.FunctionSpace)
        self.N_dofs = np.size(self.TrialFunction.vector()[:])

        # create SubDomain classes for the tabs
        negativetab = Tab()
        negativetab.set_parameters(y_lims, z_lims, tabs["negative"])
        positivetab = Tab()
        positivetab.set_parameters(y_lims, z_lims, tabs["positive"])

        # initialize mesh function for boundary domains
        boundary_markers = dolfin.MeshFunction(
            "size_t", self.fem_mesh, self.fem_mesh.topology().dim() - 1
        )
        boundary_markers.set_all(0)
        negativetab.mark(boundary_markers, 1)
        positivetab.mark(boundary_markers, 2)

        # create measure of parts of the boundary
        self.ds = dolfin.Measure(
            "ds", domain=self.fem_mesh, subdomain_data=boundary_markers
        )

        # create measure for domain
        self.dx = dolfin.dx


class Tab(dolfin.SubDomain):
    """
    A class to generate the subdomains for the tabs.
    """

    def __init__(self, y_lims, z_lims, tab):
        self.l_y = y_lims[1]
        self.l_z = z_lims[1]
        self.tab_location = [tab["y_centre"], tab["z_centre"]]
        self.tab_width = tab["width"]

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
