#
# Equation classes for the current collector
#
import pybamm

import numpy as np
import importlib

fenics_spec = importlib.util.find_spec("fenics")
if fenics_spec is not None:
    fenics = importlib.util.module_from_spec(fenics_spec)
    fenics_spec.loader.exec_module(fenics)


class Ohm(pybamm.SubModel):
    """
    Ohm's law + conservation of current for the current in the current collector.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def create_mesh(self, Ny=32, Nz=32, degree=1):
        """
        Sets up the mesh, function space etc. for the voltage problem in the
        current collectors.

        Parameters
        ----------
        Ny: int
            Number of mesh points in the y direction.
        Nz: int
            Number of mesh points in the z direction.
        degree: int
            Degree of polynomial used in FEM.
        """
        param = self.set_of_parameters
        self.Ny = Ny
        self.Nz = Nz
        self.degree = degree

        # create mesh and function space
        self.mesh = fenics.RectangleMesh(
            fenics.Point(0, 0), fenics.Point(param.Ly, 1), self.Ny, self.Nz
        )
        self.element = fenics.FunctionSpace(self.mesh, "Lagrange", self.degree)

        self.TrialFunction = fenics.TrialFunction(self.element)
        self.TestFunction = fenics.TestFunction(self.element)

        # create SubDomain classes for the tabs
        negativetab = Tab()
        negativetab.set_parameters(param, "negative")
        positivetab = Tab()
        positivetab.set_parameters(param, "positive")

        # initialize mesh function for boundary domains
        boundary_markers = fenics.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        boundary_markers.set_all(0)
        negativetab.mark(boundary_markers, 1)
        positivetab.mark(boundary_markers, 2)

        # create measure of parts of the boundary
        self.ds = fenics.Measure(
            "ds", domain=self.mesh, subdomain_data=boundary_markers
        )

        # boundary values
        self.dVdn_negativetab = fenics.Constant(
            -param.I_typ
            / (param.sigma_cn * param.aspect_ratio ** 2 * param.l_tab_n * param.l_cn)
        )
        self.dVdn_positivetab = fenics.Constant(
            -param.I_typ
            / (param.sigma_cp * param.aspect_ratio ** 2 * param.l_tab_p * param.l_cp)
        )

    def assemble(self):
        " Assemble mass and stiffness matrices, and boundary load vector."
        # create mass matrix
        M_form = self.TrialFunction * self.TestFunction * fenics.dx
        self.mass = fenics.assemble(M_form)

        # create stifnness matrix
        K_form = (
            fenics.inner(
                fenics.grad(self.TrialFunction), fenics.grad(self.TestFunction)
            )
            * fenics.dx
        )
        self.stiffness = fenics.assemble(K_form)

        # create load vectors for tabs
        neg_tab_form = self.dVdn_negativetab * self.TestFunction * self.ds(1)
        pos_tab_form = self.dVdn_positivetab * self.TestFunction * self.ds(2)
        self.load_tab_n = fenics.assemble(neg_tab_form).get_local()[:]
        self.load_tab_p = fenics.assemble(pos_tab_form).get_local()[:]

        # set functions for V, I and rhs
        self.voltage = fenics.Function(self.element)
        self.current = fenics.Function(self.element)
        self.rhs = fenics.Function(self.element)

        # number of degrees of freedom
        self.N_dofs = np.size(self.voltage.vector()[:])

        # placeholder for voltage difference
        self.voltage_difference = 1

    def solve(self):
        "Solve the linear system K*V = b(I)"
        # TO DISCUSS: at the moment we have pure Neumann BCs so need to adjust the solve
        # We are solving the problem iteratively i.e. solve K*V = b(I) then find
        # I, loop until converged, then step forward in time.
        # One way could be to solve K*V_new + M*V_new = b(I) + M*V_old (i.e. use part
        # of the old iterate). As long as you can write an OK initial guess then
        # this seems to work.

        # Store old values for error computation
        voltage_prev = self.voltage

        # Right hand side (uses the value of the current computed using the
        # previous iterate of V)
        self.rhs.vector()[:] = (
            self.load_tab_n
            + self.load_tab_p
            + np.dot(self.mass.array(), self.alpha * self.current.vector()[:])
        )

        # Solve K*V = b(I)
        # fenics.solve(self.stiffness, self.voltage.vector(), self.rhs.vector())

        # Solve K*V_new + M*V_new = b(I) + M*V_prev
        self.rhs.vector()[:] += np.dot(self.mass.array(), voltage_prev.vector()[:])
        fenics.solve(
            self.stiffness + self.mass, self.voltage.vector(), self.rhs.vector()
        )

        # Update difference in solution
        self.voltage_difference = np.linalg.norm(
            voltage_prev.vector()[:] - self.voltage.vector()[:]
        )

    def update_current(self, current):
        "Update the entries of the through-cell current density."
        self.current.vector()[:] = current

    def get_voltage(self):
        " Returns the voltage as an array"
        return self.voltage.vector()[:]


class Tab(fenics.SubDomain):
    def set_parameters(self, param, domain):
        # Set paramaters so they can be accessed from the fenics inside method
        self.l_y = param.l_y
        self.l_z = param.l_z
        if domain == "negative":
            self.tab_location = [param.centre_y_tab_n, param.centre_z_tab_n]
            self.tab_width = param.l_tab_n
        elif domain == "positive":
            self.tab_location = [param.centre_y_tab_p, param.centre_z_tab_p]
            self.tab_width = param.l_tab_p
        else:
            raise pybamm.ModelError("tab domain must be one of negative or positive")

    def inside(self, x, on_boundary):
        if fenics.near(self.tab_location[1], self.l_z):
            # tab on top
            return fenics.near(x[1], self.l_z) and fenics.between(
                x[0],
                (
                    self.tab_location[0] - self.tab_width / 2,
                    self.tab_location[0] + self.tab_width / 2,
                ),
            )
        elif fenics.near(self.tab_location[1], 0.0):
            # tab on bottom
            return fenics.near(x[1], 0.0) and fenics.between(
                x[0],
                (
                    self.tab_location[0] - self.tab_width / 2,
                    self.tab_location[0] + self.tab_width / 2,
                ),
            )
        elif fenics.near(self.tab_location[0], 0.0):
            # tab on left
            return fenics.near(x[0], 0.0) and fenics.between(
                x[1],
                (
                    self.tab_location[1] - self.tab_width / 2,
                    self.tab_location[1] + self.tab_width / 2,
                ),
            )
        elif fenics.near(self.tab_location[0], self.l_y):
            # tab on right
            return fenics.near(x[0], self.l_y) and fenics.between(
                x[1],
                (
                    self.tab_location[1] - self.tab_width / 2,
                    self.tab_location[1] + self.tab_width / 2,
                ),
            )
        else:
            raise pybamm.ModelError("tab location not valid")
