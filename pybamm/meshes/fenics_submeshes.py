#
# fenics meshes for use in PyBaMM
#
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
        domain : dict
            A dictionary that contains the limits of the spatial variables
        npts : dict
            A dictionary that contains the number of points to be used on each
            spatial variable
        """

    def __init__(self, lims, npts, degree=1, coord_sys):
        y_lims = lims["y"]
        z_lims = lims["z"]
        self.Ny = npts[]
        self.Nz = npts[]
        self.degree = degree
        self.coord_sys = coord_sys

        # create mesh and function space
        self.mesh = dolfin.RectangleMesh(
            dolfin.Point(y_lims[0], z_lims[0]), dolfin.Point(y_lims[1], z_lims[1]), self.Ny, self.Nz
        )
        self.FunctionSpace = dolfin.FunctionSpace(self.mesh, "Lagrange", self.degree)

        self.TrialFunction = dolfin.TrialFunction(self.FunctionSpace)
        self.TestFunction = dolfin.TestFunction(self.FunctionSpace)

        # create SubDomain classes for the tabs
        # TO DO: fix passing of params to Tab class
        negativetab = Tab()
        negativetab.set_parameters(param, param_vals, "negative")
        positivetab = Tab()
        positivetab.set_parameters(param, param_vals, "positive")

        # initialize mesh function for boundary domains
        boundary_markers = dolfin.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        boundary_markers.set_all(0)
        negativetab.mark(boundary_markers, 1)
        positivetab.mark(boundary_markers, 2)

        # create measure of parts of the boundary
        self.ds = dolfin.Measure(
            "ds", domain=self.mesh, subdomain_data=boundary_markers
        )


class Tab(dolfin.SubDomain):
    def set_parameters(self, param, param_vals, domain):
        # Set paramaters so they can be accessed from the dolfin inside method
        self.l_y = param_vals.process_symbol(param.l_y).evaluate(0, 0)
        self.l_z = param_vals.process_symbol(param.l_z).evaluate(0, 0)
        if domain == "negative":
            self.tab_location = [
                param_vals.process_symbol(param.centre_y_tab_n).evaluate(0, 0),
                param_vals.process_symbol(param.centre_z_tab_n).evaluate(0, 0),
            ]
            self.tab_width = param_vals.process_symbol(param.l_tab_n).evaluate(0, 0)
        elif domain == "positive":
            self.tab_location = [
                param_vals.process_symbol(param.centre_y_tab_p).evaluate(0, 0),
                param_vals.process_symbol(param.centre_z_tab_p).evaluate(0, 0),
            ]
            self.tab_width = param_vals.process_symbol(param.l_tab_p).evaluate(0, 0)
        else:
            raise pybamm.ModelError("tab domain must be one of negative or positive")

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
