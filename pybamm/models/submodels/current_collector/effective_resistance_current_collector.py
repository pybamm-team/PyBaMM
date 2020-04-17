#
# Class for calcuting the effective resistance of two-dimensional current collectors
#
import pybamm


class EffectiveResistance1D(pybamm.BaseModel):
    """A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity.
    Note:  This submodel should be solved before a one-dimensional model to calculate
    and return the effective current collector resistance.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self):
        super().__init__()
        self.name = "Effective resistance in current collector model (2D)"
        self.param = pybamm.standard_parameters_lithium_ion

        # Get necessary parameters
        param = self.param
        l_cn = param.l_cn
        l_cp = param.l_cp
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        delta = param.delta  # aspect ratio

        # Set model variables
        R_cn = pybamm.Variable(
            "Negative current collector resistance", domain="current collector"
        )
        R_cp = pybamm.Variable(
            "positive current collector resistance", domain="current collector"
        )

        # We solve using a scaled version that is better conditioned
        R_cn_scaled = R_cn * l_cn * sigma_cn_dbl_prime / delta
        R_cp_scaled = R_cp * l_cp * sigma_cp_dbl_prime / delta

        # Governing equations
        self.algebraic = {
            R_cn: pybamm.laplacian(R_cn_scaled) - 1,
            R_cp: pybamm.laplacian(R_cp_scaled) - 1,
        }

        # Boundary conditons
        self.boundary_conditions = {
            R_cn_scaled: {
                "negative tab": (0, "Dirichlet"),
                "positive tab": (0, "Neumann"),
                "no tab": (0, "Neumann"),
            },
            R_cp_scaled: {
                "negative tab": (0, "Neumann"),
                "positive tab": (0, "Dirichlet"),
                "no tab": (0, "Neumann"),
            },
        }

        # "Initial conditions" provides initial guess for solver
        self.initial_conditions = {R_cn: pybamm.Scalar(0), R_cp: pybamm.Scalar(0)}

        # Define effective current collector resistance
        R_cc_n = -pybamm.z_average(R_cn)
        R_cc_p = -pybamm.z_average(R_cp)
        R_cc = R_cc_n + R_cc_p
        R_scale = param.potential_scale / param.I_typ

        self.variables = {
            "Negative current collector resistance": R_cn,
            "Negative current collector resistance [Ohm]": R_cn * R_scale,
            "Positive current collector resistance": R_cp,
            "Positive current collector resistance [Ohm]": R_cp * R_scale,
            "Effective current collector resistance": R_cc,
            "Effective current collector resistance [Ohm]": R_cc * R_scale,
            "Effective negative current collector resistance": R_cc_n,
            "Effective negative current collector resistance [Ohm]": R_cc_n * R_scale,
            "Effective positive current collector resistance": R_cc_p,
            "Effective positive current collector resistance [Ohm]": R_cc_p * R_scale,
        }

    def get_processed_potentials(self, solution, mesh, param_values, V_av, I_av):
        """
        Calculates the potentials in the current collector given
        the average voltage and current.
        Note: This takes in the *processed* V_av and I_av from a 1D simulation
        representing the average cell behaviour and returns a dictionary of
        processed potentials.
        """
        # Process resistances
        R_cn = pybamm.ProcessedVariable(
            self.variables["Negative current collector resistance"],
            solution.t,
            solution.y,
            mesh,
        )
        R_cp = pybamm.ProcessedVariable(
            self.variables["Positive current collector resistance"],
            solution.t,
            solution.y,
            mesh,
        )

        # Create callable combination of ProcessedVariable objects for potentials
        def phi_s_cn(t, z):
            return R_cn(z=z) * I_av(t=t)

        def phi_s_cp(t, z):
            return V_av(t=t) - R_cp(z=z) * I_av(t=t)

        def V_cc(t, z):
            return phi_s_cp(t, z) - phi_s_cn(t, z)

        param = self.param
        pot_scale = param_values.evaluate(param.potential_scale)
        U_ref = param_values.evaluate(param.U_p_ref - param.U_n_ref)

        def phi_s_cn_dim(t, z):
            return phi_s_cn(t, z) * pot_scale

        def phi_s_cp_dim(t, z):
            return U_ref + phi_s_cp(t, z) * pot_scale

        def V_cc_dim(t, z):
            return U_ref + V_cc(t, z) * pot_scale

        potentials = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn_dim,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": phi_s_cp_dim,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": V_cc_dim,
        }
        return potentials

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D current collector")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.z: 32}

    @property
    def default_submesh_types(self):
        return {"current collector": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}

    @property
    def default_spatial_methods(self):
        return {"current collector": pybamm.FiniteVolume()}

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver()


class EffectiveResistance2D(pybamm.BaseModel):
    """A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity. This formulation
    assumes uniform potential across the tabs.
    Note:  This submodel should be solved before a one-dimensional model to calculate
    and return the effective current collector resistance.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self):
        super().__init__()
        self.name = "Effective resistance in current collector model (2D)"
        self.param = pybamm.standard_parameters_lithium_ion

        # Get necessary parameters
        param = self.param
        l_cn = param.l_cn
        l_cp = param.l_cp
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        delta = param.delta  # aspect ratio

        # Set model variables
        R_cn = pybamm.Variable(
            "Negative current collector resistance", domain="current collector"
        )
        R_cp = pybamm.Variable(
            "positive current collector resistance", domain="current collector"
        )

        # We solve using a scaled version that is better conditioned
        R_cn_scaled = R_cn * l_cn * sigma_cn_dbl_prime / delta
        R_cp_scaled = R_cp * l_cp * sigma_cp_dbl_prime / delta

        # Governing equations
        self.algebraic = {
            R_cn: pybamm.laplacian(R_cn_scaled) - pybamm.source(1, R_cn_scaled),
            R_cp: pybamm.laplacian(R_cp_scaled) - pybamm.source(1, R_cp_scaled),
        }

        # Boundary conditons
        self.boundary_conditions = {
            R_cn_scaled: {
                "negative tab": (0, "Dirichlet"),
                "positive tab": (0, "Neumann"),
            },
            R_cp_scaled: {
                "negative tab": (0, "Neumann"),
                "positive tab": (0, "Dirichlet"),
            },
        }

        # "Initial conditions" provides initial guess for solver
        self.initial_conditions = {R_cn: pybamm.Scalar(0), R_cp: pybamm.Scalar(0)}

        # Define effective current collector resistance
        R_cc_n = -pybamm.yz_average(R_cn)
        R_cc_p = -pybamm.yz_average(R_cp)
        R_cc = R_cc_n + R_cc_p
        R_scale = param.potential_scale / param.I_typ

        self.variables = {
            "Negative current collector resistance": R_cn,
            "Negative current collector resistance [Ohm]": R_cn * R_scale,
            "Positive current collector resistance": R_cp,
            "Positive current collector resistance [Ohm]": R_cp * R_scale,
            "Effective current collector resistance": R_cc,
            "Effective current collector resistance [Ohm]": R_cc * R_scale,
            "Effective negative current collector resistance": R_cc_n,
            "Effective negative current collector resistance [Ohm]": R_cc_n * R_scale,
            "Effective positive current collector resistance": R_cc_p,
            "Effective positive current collector resistance [Ohm]": R_cc_p * R_scale,
        }

    def get_processed_potentials(self, solution, mesh, param_values, V_av, I_av):
        """
        Calculates the potentials in the current collector given
        the average voltage and current.
        Note: This takes in the *processed* V_av and I_av from a 1D simulation
        representing the average cell behaviour and returns a dictionary of
        processed potentials.
        """
        # Process resistances
        R_cn = pybamm.ProcessedVariable(
            self.variables["Negative current collector resistance"],
            solution.t,
            solution.y,
            mesh,
        )
        R_cp = pybamm.ProcessedVariable(
            self.variables["Positive current collector resistance"],
            solution.t,
            solution.y,
            mesh,
        )

        # Create callable combination of ProcessedVariable objects for potentials
        def phi_s_cn(t, y, z):
            return R_cn(y=y, z=z) * I_av(t=t)

        def phi_s_cp(t, y, z):
            return V_av(t=t) - R_cp(y=y, z=z) * I_av(t=t)

        def V_cc(t, y, z):
            return phi_s_cp(t, y, z) - phi_s_cn(t, y, z)

        param = self.param
        pot_scale = param_values.evaluate(param.potential_scale)
        U_ref = param_values.evaluate(param.U_p_ref - param.U_n_ref)

        def phi_s_cn_dim(t, y, z):
            return phi_s_cn(t, y, z) * pot_scale

        def phi_s_cp_dim(t, y, z):
            return U_ref + phi_s_cp(t, y, z) * pot_scale

        def V_cc_dim(t, y, z):
            return U_ref + V_cc(t, y, z) * pot_scale

        potentials = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn_dim,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": phi_s_cp_dim,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": V_cc_dim,
        }
        return potentials

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        return pybamm.Geometry("2D current collector")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.y: 32, var.z: 32}

    @property
    def default_submesh_types(self):
        return {
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh)
        }

    @property
    def default_spatial_methods(self):
        return {"current collector": pybamm.ScikitFiniteElement()}

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver()


class AlternativeEffectiveResistance2D(pybamm.BaseModel):
    """A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity. Uses an alternative
    formulation which assumes uniform current density across the positive tab rather
    than uniform potential. Note that the variables in this formulation are also
    defined slightly differently.
    Note:  This submodel should be solved before a one-dimensional model to calculate
    and return the effective current collector resistance.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self):
        super().__init__()
        self.name = "Effective resistance in current collector model (2D)"
        self.param = pybamm.standard_parameters_lithium_ion

        # Get necessary parameters
        param = self.param
        l_y = param.l_y
        l_z = param.l_z
        l_cn = param.l_cn
        l_cp = param.l_cp
        l_tab_p = param.l_tab_p
        A_tab_p = l_cp * l_tab_p
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        delta = param.delta

        # Set model variables -- we solve a unit problem in each current collector
        # then relate this to the potentials and resistances later
        f_n = pybamm.Variable(
            "Unit solution in negative current collector", domain="current collector"
        )
        f_p = pybamm.Variable(
            "Unit solution in positive current collector", domain="current collector"
        )

        # Governing equations -- we impose that the average of f_p is zero
        # by introducing a Lagrange multiplier
        c = pybamm.Variable("Lagrange multiplier")

        self.algebraic = {
            f_n: pybamm.laplacian(f_n) + pybamm.source(1, f_n),
            f_p: pybamm.laplacian(f_p)
            - pybamm.source(1, f_p)
            + c * pybamm.DefiniteIntegralVector(f_p, vector_type="column"),
            c: pybamm.yz_average(f_p) + 0 * c,
        }

        # Boundary conditons
        pos_tab_bc = l_y * l_z * l_cp / A_tab_p
        self.boundary_conditions = {
            f_n: {"negative tab": (0, "Dirichlet"), "positive tab": (0, "Neumann")},
            f_p: {
                "negative tab": (0, "Neumann"),
                "positive tab": (pos_tab_bc, "Neumann"),
            },
        }

        # "Initial conditions" provides initial guess for solver
        self.initial_conditions = {
            f_n: pybamm.Scalar(0),
            f_p: pybamm.Scalar(0),
            c: pybamm.Scalar(0),
        }

        # Define effective current collector resistance
        R_cc_n = (
            delta * pybamm.yz_average(f_n) / (l_y * l_z * l_cn * sigma_cn_dbl_prime)
        )
        R_cc_p = (
            delta
            * pybamm.BoundaryIntegral(f_p, "positive tab")
            / (l_y * l_z * l_cp * sigma_cp_dbl_prime)
        )
        R_cc = R_cc_n + R_cc_p
        R_scale = param.potential_scale / param.I_typ

        self.variables = {
            "Unit solution in negative current collector": f_n,
            "Unit solution in positive current collector": f_p,
            "Effective current collector resistance": R_cc,
            "Effective current collector resistance [Ohm]": R_cc * R_scale,
            "Effective negative current collector resistance": R_cc_n,
            "Effective negative current collector resistance [Ohm]": R_cc_n * R_scale,
            "Effective positive current collector resistance": R_cc_p,
            "Effective positive current collector resistance [Ohm]": R_cc_p * R_scale,
        }

    def get_processed_potentials(self, solution, mesh, param_values, V_av, I_av):
        """
        Calculates the potentials in the current collector given
        the average voltage and current.
        Note: This takes in the *processed* V_av and I_av from a 1D simulation
        representing the average cell behaviour and returns a dictionary of
        processed potentials.
        """
        # Get parameters
        param = self.param
        delta = param_values.evaluate(param.delta)
        l_cn = param_values.evaluate(param.l_cn)
        l_cp = param_values.evaluate(param.l_cp)
        sigma_cn_dbl_prime = param_values.evaluate(param.sigma_cn_dbl_prime)
        sigma_cp_dbl_prime = param_values.evaluate(param.sigma_cp_dbl_prime)
        pot_scale = param_values.evaluate(param.potential_scale)
        U_ref = param_values.evaluate(param.U_p_ref - param.U_n_ref)

        # Process unit solutions
        f_n = pybamm.ProcessedVariable(
            self.variables["Unit solution in negative current collector"],
            solution.t,
            solution.y,
            mesh,
        )
        f_p = pybamm.ProcessedVariable(
            self.variables["Unit solution in positive current collector"],
            solution.t,
            solution.y,
            mesh,
        )

        # Create callable combination of ProcessedVariable objects for potentials
        def phi_s_cn(t, y, z):
            return -(delta * I_av(t=t) / l_cn / sigma_cn_dbl_prime) * f_n(y=y, z=z)

        def phi_s_cp(t, y, z):
            return V_av(t=t) + (delta * I_av(t=t) / l_cp / sigma_cp_dbl_prime) * f_p(
                y=y, z=z
            )

        def V_cc(t, y, z):
            return phi_s_cp(t, y, z) - phi_s_cn(t, y, z)

        def phi_s_cn_dim(t, y, z):
            return phi_s_cn(t, y, z) * pot_scale

        def phi_s_cp_dim(t, y, z):
            return U_ref + phi_s_cp(t, y, z) * pot_scale

        def V_cc_dim(t, y, z):
            return U_ref + V_cc(t, y, z) * pot_scale

        potentials = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn_dim,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": phi_s_cp_dim,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": V_cc_dim,
        }
        return potentials

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        return pybamm.Geometry("2D current collector")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.y: 32, var.z: 32}

    @property
    def default_submesh_types(self):
        return {
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh)
        }

    @property
    def default_spatial_methods(self):
        return {"current collector": pybamm.ScikitFiniteElement()}

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver()
