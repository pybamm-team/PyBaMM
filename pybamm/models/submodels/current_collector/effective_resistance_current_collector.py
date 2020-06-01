#
# Classes for calcuting the effective resistance of current collectors in a pouch cell
#
import pybamm


class EffectiveResistance(pybamm.BaseModel):
    """
    A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity. For details see [1]_.
    Note that this formulation assumes uniform *potential* across the tabs. See
    :class:`pybamm.AlternativeEffectiveResistance2D` for the formulation that
    assumes a uniform *current density* at the tabs (in 1D the two formulations
    are equivalent).

    Parameters
    ----------
    options: dict
        A dictionary of options to be passed to the model. The options that can
        be set are listed below.

            * "dimensionality" : int, optional
                Sets the dimension of the current collector problem. Can be 1
                (default) or 2.
    name : str, optional
        The name of the model.

    References
    ----------
    .. [1] R Timms, SG Marquis, V Sulzer, CP Please and SJ Chapman. “Asymptotic
           Reduction of a Lithium-ion Pouch Cell Model”. Submitted, 2020.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(
        self, options=None, name="Effective resistance in current collector model"
    ):
        super().__init__(name)
        self.options = options
        self.param = pybamm.standard_parameters_lithium_ion

        self.variables = self.get_fundamental_variables()
        self.set_algebraic(self.variables)
        self.set_boundary_conditions(self.variables)
        self.set_initial_conditions(self.variables)

        pybamm.citations.register("timms2020")

    def get_fundamental_variables(self):
        # Get necessary parameters
        param = self.param
        l_cn = param.l_cn
        l_cp = param.l_cp
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        delta = param.delta  # aspect ratio

        # Set model variables: Note: we solve using a scaled version that is
        # better conditioned
        R_cn_scaled = pybamm.Variable(
            "Scaled negative current collector resistance", domain="current collector"
        )
        R_cp_scaled = pybamm.Variable(
            "Scaled positive current collector resistance", domain="current collector"
        )
        R_cn = delta * R_cn_scaled / (l_cn * sigma_cn_dbl_prime)
        R_cp = delta * R_cp_scaled / (l_cp * sigma_cp_dbl_prime)

        # Define effective current collector resistance
        if self.options["dimensionality"] == 1:
            R_cc_n = -pybamm.z_average(R_cn)
            R_cc_p = -pybamm.z_average(R_cp)
        elif self.options["dimensionality"] == 2:
            R_cc_n = -pybamm.yz_average(R_cn)
            R_cc_p = -pybamm.yz_average(R_cp)
        R_cc = R_cc_n + R_cc_p
        R_scale = param.potential_scale / param.I_typ

        variables = {
            "Scaled negative current collector resistance": R_cn_scaled,
            "Negative current collector resistance": R_cn,
            "Negative current collector resistance [Ohm]": R_cn * R_scale,
            "Scaled positive current collector resistance": R_cp_scaled,
            "Positive current collector resistance": R_cp,
            "Positive current collector resistance [Ohm]": R_cp * R_scale,
            "Effective current collector resistance": R_cc,
            "Effective current collector resistance [Ohm]": R_cc * R_scale,
            "Effective negative current collector resistance": R_cc_n,
            "Effective negative current collector resistance [Ohm]": R_cc_n * R_scale,
            "Effective positive current collector resistance": R_cc_p,
            "Effective positive current collector resistance [Ohm]": R_cc_p * R_scale,
        }

        # Add spatial variables
        var = pybamm.standard_spatial_vars
        L_y = pybamm.geometric_parameters.L_y
        L_z = pybamm.geometric_parameters.L_z
        if self.options["dimensionality"] == 1:
            variables.update({"z": var.z, "z [m]": var.z * L_z})
        elif self.options["dimensionality"] == 2:
            variables.update(
                {"y": var.y, "y [m]": var.y * L_y, "z": var.z, "z [m]": var.z * L_z}
            )

        return variables

    def set_algebraic(self, variables):
        R_cn_scaled = variables["Scaled negative current collector resistance"]
        R_cp_scaled = variables["Scaled positive current collector resistance"]
        self.algebraic = {
            R_cn_scaled: pybamm.laplacian(R_cn_scaled) - pybamm.source(1, R_cn_scaled),
            R_cp_scaled: pybamm.laplacian(R_cp_scaled) - pybamm.source(1, R_cp_scaled),
        }

    def set_boundary_conditions(self, variables):
        R_cn_scaled = variables["Scaled negative current collector resistance"]
        R_cp_scaled = variables["Scaled positive current collector resistance"]

        if self.options["dimensionality"] == 1:
            self.boundary_conditions = {
                R_cn_scaled: {
                    "negative tab": (0, "Dirichlet"),
                    "no tab": (0, "Neumann"),
                },
                R_cp_scaled: {
                    "positive tab": (0, "Dirichlet"),
                    "no tab": (0, "Neumann"),
                },
            }
        elif self.options["dimensionality"] == 2:
            self.boundary_conditions = {
                R_cn_scaled: {
                    "negative tab": (0, "Dirichlet"),
                    "positive tab": (0, "Neumann"),
                },
                R_cp_scaled: {
                    "positive tab": (0, "Dirichlet"),
                    "negative tab": (0, "Neumann"),
                },
            }

    def set_initial_conditions(self, variables):
        R_cn_scaled = variables["Scaled negative current collector resistance"]
        R_cp_scaled = variables["Scaled positive current collector resistance"]
        self.initial_conditions = {
            R_cn_scaled: pybamm.Scalar(0),
            R_cp_scaled: pybamm.Scalar(0),
        }

    def post_process(self, solution, param_values, V_av, I_av):
        """
        Calculates the potentials in the current collector and the terminal
        voltage given the average voltage and current.
        Note: This takes in the *processed* V_av and I_av from a 1D simulation
        representing the average cell behaviour and returns a dictionary of
        processed potentials.
        """
        param = self.param
        pot_scale = param_values.evaluate(param.potential_scale)
        U_ref = param_values.evaluate(param.U_p_ref - param.U_n_ref)

        # Process resistances
        R_cn = solution["Negative current collector resistance"]
        R_cp = solution["Positive current collector resistance"]
        R_cc = solution["Effective current collector resistance"]

        # Create callable combination of ProcessedVariable objects for potentials
        # and terminal voltage
        def V(t):
            "Account for effective current collector resistance"
            return V_av(t) - I_av(t) * R_cc(t)

        def phi_s_cn(t, z, y=None):
            return R_cn(y=y, z=z) * I_av(t=t)

        def phi_s_cp(t, z, y=None):
            return V(t) - R_cp(y=y, z=z) * I_av(t=t)

        def V_cc(t, z, y=None):
            return phi_s_cp(t, z, y=y) - phi_s_cn(t, z, y=y)

        def V_dim(t):
            return U_ref + V(t) * pot_scale

        def phi_s_cn_dim(t, z, y=None):
            return phi_s_cn(t, z, y=y) * pot_scale

        def phi_s_cp_dim(t, z, y=None):
            return U_ref + phi_s_cp(t, z, y=y) * pot_scale

        def V_cc_dim(t, z, y=None):
            return U_ref + V_cc(t, z, y=y) * pot_scale

        processed_vars = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn_dim,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": phi_s_cp_dim,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": V_cc_dim,
            "Terminal voltage": V,
            "Terminal voltage [V]": V_dim,
        }
        return processed_vars

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        geometry = {}
        var = pybamm.standard_spatial_vars
        if self.options["dimensionality"] == 1:
            geometry["current collector"] = {
                var.z: {"min": 0, "max": 1},
                "tabs": {
                    "negative": {
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_n
                    },
                    "positive": {
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_p
                    },
                },
            }
        elif self.options["dimensionality"] == 2:
            geometry["current collector"] = {
                var.y: {"min": 0, "max": pybamm.geometric_parameters.l_y},
                var.z: {"min": 0, "max": pybamm.geometric_parameters.l_z},
                "tabs": {
                    "negative": {
                        "y_centre": pybamm.geometric_parameters.centre_y_tab_n,
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_n,
                        "width": pybamm.geometric_parameters.l_tab_n,
                    },
                    "positive": {
                        "y_centre": pybamm.geometric_parameters.centre_y_tab_p,
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_p,
                        "width": pybamm.geometric_parameters.l_tab_p,
                    },
                },
            }
        return pybamm.Geometry(geometry)

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.y: 32, var.z: 32}

    @property
    def default_submesh_types(self):
        if self.options["dimensionality"] == 1:
            return {"current collector": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)}
        elif self.options["dimensionality"] == 2:
            return {
                "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh)
            }

    @property
    def default_spatial_methods(self):
        if self.options["dimensionality"] == 1:
            return {"current collector": pybamm.FiniteVolume()}
        elif self.options["dimensionality"] == 2:
            return {"current collector": pybamm.ScikitFiniteElement()}

    @property
    def default_solver(self):
        return pybamm.CasadiAlgebraicSolver()

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        default_options = {"dimensionality": 1}
        extra_options = extra_options or {}

        options = pybamm.FuzzyDict(default_options)
        # any extra options overwrite the default options
        for name, opt in extra_options.items():
            if name in default_options:
                options[name] = opt
            else:
                raise pybamm.OptionError(
                    "Option '{}' not recognised. Best matches are {}".format(
                        name, options.get_best_matches(name)
                    )
                )

        if options["dimensionality"] not in [1, 2]:
            raise pybamm.OptionError(
                "Dimension of current collectors must be 1 or 2, not {}".format(
                    options["dimensionality"]
                )
            )
        self._options = options


class AlternativeEffectiveResistance2D(pybamm.BaseModel):
    """
    A model which calculates the effective Ohmic resistance of the 2D current
    collectors in the limit of large electrical conductivity. This model assumes
    a uniform *current density* at the tabs and the solution is computed by first
    solving and auxilliary problem which is the related to the resistances.

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
        l_tab_p = param.l_tab_p
        A_tab_p = l_cp * l_tab_p
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        delta = param.delta

        # Set model variables -- we solve a auxilliary problem in each current collector
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
            c: pybamm.laplacian(f_p)
            - pybamm.source(1, f_p)
            + c * pybamm.DefiniteIntegralVector(f_p, vector_type="column"),
            f_p: pybamm.yz_average(f_p) + 0 * c,
        }

        # Boundary conditons
        pos_tab_bc = l_cp / A_tab_p
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
        R_cc_n = delta * pybamm.yz_average(f_n) / (l_cn * sigma_cn_dbl_prime)
        R_cc_p = (
            delta
            * pybamm.BoundaryIntegral(f_p, "positive tab")
            / (l_cp * sigma_cp_dbl_prime)
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

        # Add spatial variables
        var = pybamm.standard_spatial_vars
        L_y = pybamm.geometric_parameters.L_y
        L_z = pybamm.geometric_parameters.L_z
        self.variables.update(
            {"y": var.y, "y [m]": var.y * L_y, "z": var.z, "z [m]": var.z * L_z}
        )

        pybamm.citations.register("timms2020")

    def post_process(self, solution, param_values, V_av, I_av):
        """
        Calculates the potentials in the current collector given
        the average voltage and current.
        Note: This takes in the *processed* V_av and I_av from a 1D simulation
        representing the average cell behaviour and returns a dictionary of
        processed potentials.
        """
        # Get evaluated parameters
        param = self.param
        delta = param_values.evaluate(param.delta)
        l_cn = param_values.evaluate(param.l_cn)
        l_cp = param_values.evaluate(param.l_cp)
        sigma_cn_dbl_prime = param_values.evaluate(param.sigma_cn_dbl_prime)
        sigma_cp_dbl_prime = param_values.evaluate(param.sigma_cp_dbl_prime)
        pot_scale = param_values.evaluate(param.potential_scale)
        U_ref = param_values.evaluate(param.U_p_ref - param.U_n_ref)

        # Process unit solutions
        f_n = solution["Unit solution in negative current collector"]
        f_p = solution["Unit solution in positive current collector"]

        # Get effective resistance
        R_cc = solution["Effective current collector resistance"]

        # Create callable combination of ProcessedVariable objects for potentials
        def V(t):
            "Account for effective current collector resistance"
            return V_av(t) - I_av(t) * R_cc(t)

        def phi_s_cn(t, y, z):
            return -(delta * I_av(t=t) / l_cn / sigma_cn_dbl_prime) * f_n(y=y, z=z)

        def phi_s_cp(t, y, z):
            return V(t) + (delta * I_av(t=t) / l_cp / sigma_cp_dbl_prime) * f_p(
                y=y, z=z
            )

        def V_cc(t, y, z):
            return phi_s_cp(t, y, z) - phi_s_cn(t, y, z)

        def V_dim(t):
            return U_ref + V(t) * pot_scale

        def phi_s_cn_dim(t, y, z):
            return phi_s_cn(t, y, z) * pot_scale

        def phi_s_cp_dim(t, y, z):
            return U_ref + phi_s_cp(t, y, z) * pot_scale

        def V_cc_dim(t, y, z):
            return U_ref + V_cc(t, y, z) * pot_scale

        processed_vars = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn_dim,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": phi_s_cp_dim,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": V_cc_dim,
            "Terminal voltage": V,
            "Terminal voltage [V]": V_dim,
        }
        return processed_vars

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        var = pybamm.standard_spatial_vars
        geometry = {
            "current collector": {
                var.y: {"min": 0, "max": pybamm.geometric_parameters.l_y},
                var.z: {"min": 0, "max": pybamm.geometric_parameters.l_z},
                "tabs": {
                    "negative": {
                        "y_centre": pybamm.geometric_parameters.centre_y_tab_n,
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_n,
                        "width": pybamm.geometric_parameters.l_tab_n,
                    },
                    "positive": {
                        "y_centre": pybamm.geometric_parameters.centre_y_tab_p,
                        "z_centre": pybamm.geometric_parameters.centre_z_tab_p,
                        "width": pybamm.geometric_parameters.l_tab_p,
                    },
                },
            }
        }
        return pybamm.Geometry(geometry)

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
        return pybamm.CasadiAlgebraicSolver()
