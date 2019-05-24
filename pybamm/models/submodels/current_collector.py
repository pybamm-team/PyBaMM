#
# Equation classes for the current collector
#
import pybamm

import numpy as np
import importlib

dolfin_spec = importlib.util.find_spec("dolfin")
if dolfin_spec is not None:
    dolfin = importlib.util.module_from_spec(dolfin_spec)
    dolfin_spec.loader.exec_module(dolfin)


class OhmTwoDimensional(pybamm.SubModel):
    """
    Ohm's law + conservation of current for the current in the current collectors.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters, parameter_values):
        super().__init__(set_of_parameters)
        self.parameter_values = parameter_values

    def set_algebraic_system(self, v_local, i_local):
        """
        PDE system for current in the current collectors, using Ohm's law

        Parameters
        ----------
        v_local : :class:`pybamm.Variable`
            Local cell voltage
        i_local : :class:`pybamm.Variable`
            Local through-cell current density

        """
        param = self.set_of_parameters
        v_local = pybamm.Variable("Local cell voltage", domain="current collector")
        i_local = pybamm.Variable(
            "Local through-cell current density", domain="current collector"
        )

        # create finite element mesh
        # Question: to create the mesh we need evaluated parameters. I wasn't sure
        # how to just pass param here (instead of using parameter values), and only
        # have the mesh being created during discretisation? Maybe I should make
        # a new spatial method, and define pybamm.fenics_laplacian or something,
        # which then calls this during discretisation, by which point the params
        # have been procseed?
        self.create_mesh()

        # assemble finite element matrices using fenics
        # TO DO: We may want to have fewer 1D models than mesh points. The method
        # creates a coarse and fine mesh, but we currently don't use the coarse one.
        # could have i_local_fine and i_local_coarse and relate with some kind of interpolation
        # NOTE: to get vertex map to degrees of freedom use dolfin.vertex_to_dof_map(self.FunctionSpace)
        # (may be needed for indexing in pybamm to get BCs). to see coordinates use
        # self.mesh.coordinates()
        self.assemble()
        # maybe assemble should just return these, though maybe it is useful for them
        # to be an attribute of the current collector model for plotting/calculating other things after the solve
        mass = pybamm.Matrix(self.mass.array())
        stiffness = pybamm.Matrix(self.stiffness.array())
        trial_integral = pybamm.Matrix(self.trial_integral.array())

        # algebraic equations
        applied_current = param.current_with_time
        # rhs ( -grad^2(v_local) = alpha*i_local + boundary conditions)
        rhs = param.alpha * mass @ i_local + applied_current * (
            self.load_tab_n + self.load_tab_p
        )
        self.algebraic = {
            v_local: stiffness @ v_local + rhs,
            i_local: trial_integral @ i_local - applied_current,
        }
        self.initial_conditions = {
            v_local: param.U_p(param.c_p_init) - param.U_n(param.c_n_init),
            i_local: applied_current / param.l_y / param.l_z,
        }
        self.variables = {
            "Local cell voltage": v_local,
            "Local through-cell current density": i_local,
        }

    def create_mesh(self, Ny=32, Nz=32, ny=3, nz=3, degree=1):
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
        ny: int
            Number of mesh points at which 1D model will be evaluated in the y direction.
        nz: int
            Number of mesh points at which 1D model will be evaluated in the z direction.
        """
        param = self.set_of_parameters
        param_vals = self.parameter_values
        self.Ny = Ny
        self.Nz = Nz
        self.ny = ny
        self.nz = nz
        # TO DO: check method for higher degree elements
        self.degree = degree

        # create mesh and function space
        l_y = param_vals.process_symbol(param.l_y).evaluate(0, 0)
        l_z = param_vals.process_symbol(param.l_z).evaluate(0, 0)
        self.mesh = dolfin.RectangleMesh(
            dolfin.Point(0, 0), dolfin.Point(l_y, l_z), self.Ny, self.Nz
        )
        self.mesh_coarse = dolfin.RectangleMesh(
            dolfin.Point(0, 0), dolfin.Point(l_y, l_z), self.ny, self.nz
        )
        self.FunctionSpace = dolfin.FunctionSpace(self.mesh, "Lagrange", self.degree)
        self.FunctionSpace_coarse = dolfin.FunctionSpace(
            self.mesh_coarse, "Lagrange", self.degree
        )

        self.TrialFunction = dolfin.TrialFunction(self.FunctionSpace)
        self.TestFunction = dolfin.TestFunction(self.FunctionSpace)

        # create SubDomain classes for the tabs
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

        # boundary values (assumes a current of 1, and is then multiplied by i_cell
        # in the algebraic equation for i_local)
        dVdn_neg = param_vals.process_symbol(
            -1
            / (
                param.sigma_cn
                * (param.L_x / param.L_z) ** 2
                * param.l_tab_n
                * param.l_cn
            )
        ).evaluate(0, 0)
        self.dVdn_negativetab = dolfin.Constant(dVdn_neg)
        dVdn_pos = param_vals.process_symbol(
            -1
            / (
                param.sigma_cp
                * (param.L_x / param.L_z) ** 2
                * param.l_tab_p
                * param.l_cp
            )
        ).evaluate(0, 0)
        self.dVdn_positivetab = dolfin.Constant(dVdn_pos)

    def assemble(self):
        " Assemble mass and stiffness matrices, and boundary load vector."
        # create mass matrix
        M_form = self.TrialFunction * self.TestFunction * dolfin.dx
        self.mass = dolfin.assemble(M_form)

        # create stifnness matrix
        K_form = (
            dolfin.inner(
                dolfin.grad(self.TrialFunction), dolfin.grad(self.TestFunction)
            )
            * dolfin.dx
        )
        self.stiffness = dolfin.assemble(K_form)

        # create quadrature matrix (for evaluating conservation of current constraint)
        # TO DO: think at the moment this submodel only works for degree 1 elements
        # (nodes and dof coincide). check for other elements
        self.trial_integral = dolfin.assemble(self.TrialFunction * dolfin.dx)

        # create load vectors for tabs
        neg_tab_form = self.dVdn_negativetab * self.TestFunction * self.ds(1)
        pos_tab_form = self.dVdn_positivetab * self.TestFunction * self.ds(2)
        self.load_tab_n = dolfin.assemble(neg_tab_form).get_local()[:]
        self.load_tab_p = dolfin.assemble(pos_tab_form).get_local()[:]

        # set functions for V, I and load
        self.voltage = dolfin.Function(self.FunctionSpace)
        self.voltage_prev = dolfin.Function(self.FunctionSpace)
        self.voltage_coarse = dolfin.Function(self.FunctionSpace_coarse)

        self.current = dolfin.Function(self.FunctionSpace)
        self.current_coarse = dolfin.Function(self.FunctionSpace_coarse)
        self.load = dolfin.Function(self.FunctionSpace)

        # number of degrees of freedom
        self.N_dofs = np.size(self.voltage.vector()[:])
        self.n_dofs = np.size(self.voltage_coarse.vector()[:])


class IterativeOhmTwoDimensional(pybamm.SubModel):
    """
    Ohm's law + conservation of current for the current in the current collectors.
    NOTE:  don't think we will actually solve this way, so can probably be removed.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters, parameter_values):
        super().__init__(set_of_parameters)
        self.parameter_values = parameter_values

    def create_mesh(self, Ny=32, Nz=32, ny=3, nz=3, degree=1):
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
        ny: int
            Number of mesh points at which 1D model will be evaluated in the y direction.
        nz: int
            Number of mesh points at which 1D model will be evaluated in the z direction.
        """
        param = self.set_of_parameters
        param_vals = self.parameter_values
        self.Ny = Ny
        self.Nz = Nz
        self.ny = ny
        self.nz = nz
        self.degree = degree

        # create mesh and function space
        l_y = param_vals.process_symbol(param.l_y).evaluate(0, 0)
        l_z = param_vals.process_symbol(param.l_z).evaluate(0, 0)
        self.mesh = dolfin.RectangleMesh(
            dolfin.Point(0, 0), dolfin.Point(l_y, l_z), self.Ny, self.Nz
        )
        self.mesh_coarse = dolfin.RectangleMesh(
            dolfin.Point(0, 0), dolfin.Point(l_y, l_z), self.ny, self.nz
        )
        self.FunctionSpace = dolfin.FunctionSpace(self.mesh, "Lagrange", self.degree)
        self.FunctionSpace_coarse = dolfin.FunctionSpace(
            self.mesh_coarse, "Lagrange", self.degree
        )

        self.TrialFunction = dolfin.TrialFunction(self.FunctionSpace)
        self.TestFunction = dolfin.TestFunction(self.FunctionSpace)

        # create SubDomain classes for the tabs
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

        # boundary values
        dVdn_neg = param_vals.process_symbol(
            -1
            / (
                param.sigma_cn
                * (param.L_x / param.L_z) ** 2
                * param.l_tab_n
                * param.l_cn
            )
        ).evaluate(0, 0)
        self.dVdn_negativetab = dolfin.Constant(dVdn_neg)
        dVdn_pos = param_vals.process_symbol(
            -1
            / (
                param.sigma_cp
                * (param.L_x / param.L_z) ** 2
                * param.l_tab_p
                * param.l_cp
            )
        ).evaluate(0, 0)
        self.dVdn_positivetab = dolfin.Constant(dVdn_pos)

    def assemble(self):
        " Assemble mass and stiffness matrices, and boundary load vector."
        # create mass matrix
        M_form = self.TrialFunction * self.TestFunction * dolfin.dx
        self.mass = dolfin.assemble(M_form)

        # create stifnness matrix
        K_form = (
            dolfin.inner(
                dolfin.grad(self.TrialFunction), dolfin.grad(self.TestFunction)
            )
            * dolfin.dx
        )
        self.stiffness = dolfin.assemble(K_form)

        # create load vectors for tabs
        neg_tab_form = self.dVdn_negativetab * self.TestFunction * self.ds(1)
        pos_tab_form = self.dVdn_positivetab * self.TestFunction * self.ds(2)
        self.load_tab_n = dolfin.assemble(neg_tab_form).get_local()[:]
        self.load_tab_p = dolfin.assemble(pos_tab_form).get_local()[:]

        # set functions for V, I and load
        self.voltage = dolfin.Function(self.FunctionSpace)
        self.voltage_prev = dolfin.Function(self.FunctionSpace)
        self.voltage_coarse = dolfin.Function(self.FunctionSpace_coarse)

        self.current = dolfin.Function(self.FunctionSpace)
        self.current_coarse = dolfin.Function(self.FunctionSpace_coarse)
        self.load = dolfin.Function(self.FunctionSpace)

        # number of degrees of freedom
        self.N_dofs = np.size(self.voltage.vector()[:])
        self.n_dofs = np.size(self.voltage_coarse.vector()[:])

        # placeholder for voltage difference
        self.voltage_difference = 1

    def solve(self, t):
        "Solve the linear system K*V = b(I)"
        # TO DISCUSS: at the moment we have pure Neumann BCs so need to adjust the solve
        # We are solving the problem iteratively i.e. solve K*V = b(I) then find
        # I, loop until converged, then step forward in time.

        # store old values for error computation
        self.voltage_prev.vector()[:] = self.voltage.vector()[:]

        # adjusted right hand side := b(I) + M*V_prev
        # (uses the value of the current computed using the previous iterate of V)
        alpha = self.parameter_values.process_symbol(
            self.set_of_parameters.alpha
        ).evaluate(0, 0)
        applied_current = self.parameter_values.process_symbol(
            self.set_of_parameters.current_with_time
        ).evaluate(t, 0)
        self.load.vector()[:] = applied_current * (
            self.load_tab_n + self.load_tab_p
        ) + np.dot(
            self.mass.array(),
            alpha * self.current.vector()[:] + self.voltage_prev.vector()[:],
        )

        # solve K*V_new + M*V_new = b(I) + M*V_prev
        dolfin.solve(
            self.stiffness + self.mass, self.voltage.vector(), self.load.vector()
        )

        # Update difference in solution
        self.voltage_difference = np.linalg.norm(
            self.voltage_prev.vector()[:] - self.voltage.vector()[:]
        )

    def update_current(self, current, coarse=True):
        "Update the entries of the through-cell current density."
        if coarse:
            self.current_coarse.vector()[:] = current
            self.current = dolfin.project(self.current_coarse, self.FunctionSpace)
        else:
            self.current.vector()[:] = current

    def get_voltage(self, coarse=False):
        "Returns the voltage as an array"
        if coarse:
            self.voltage_coarse = dolfin.project(
                self.voltage, self.FunctionSpace_coarse
            )
            return self.voltage_coarse.vector()[:]
        else:
            return self.voltage.vector()[:]

    def get_current(self, coarse=False):
        "Returns the through-cell current density as an array"
        if coarse:
            return self.current_coarse.vector()[:]
        else:
            return self.current.vector()[:]


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
