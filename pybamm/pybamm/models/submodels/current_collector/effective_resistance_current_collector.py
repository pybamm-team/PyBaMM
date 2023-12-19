#
# Classes for calcuting the effective resistance of current collectors in a pouch cell
#
import pybamm


class BaseEffectiveResistance(pybamm.BaseModel):
    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues("Marquis2019")

    @property
    def default_geometry(self):
        geometry = {}
        param = self.param
        if self.options["dimensionality"] == 1:
            geometry["current collector"] = {
                "z": {"min": 0, "max": param.L_z},
                "tabs": {
                    "negative": {"z_centre": param.n.centre_z_tab},
                    "positive": {"z_centre": param.p.centre_z_tab},
                },
            }
        elif self.options["dimensionality"] == 2:
            geometry["current collector"] = {
                "y": {"min": 0, "max": param.L_y},
                "z": {"min": 0, "max": param.L_z},
                "tabs": {
                    "negative": {
                        "y_centre": param.n.centre_y_tab,
                        "z_centre": param.n.centre_z_tab,
                        "width": param.n.L_tab,
                    },
                    "positive": {
                        "y_centre": param.p.centre_y_tab,
                        "z_centre": param.p.centre_z_tab,
                        "width": param.p.L_tab,
                    },
                },
            }
        return pybamm.Geometry(geometry)

    @property
    def default_var_pts(self):
        return {"y": 32, "z": 32}

    @property
    def default_submesh_types(self):
        if self.options["dimensionality"] == 1:
            return {"current collector": pybamm.Uniform1DSubMesh}
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


class EffectiveResistance(BaseEffectiveResistance):
    """
    A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity. For details see
    :footcite:t:`Timms2021`.
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

    """

    def __init__(
        self, options=None, name="Effective resistance in current collector model"
    ):
        super().__init__(name)

        pybamm.citations.register("Timms2021")

        self.options = options
        self.param = pybamm.LithiumIonParameters()

        self.variables = self.get_fundamental_variables()
        self.set_algebraic(self.variables)
        self.set_boundary_conditions(self.variables)
        self.set_initial_conditions(self.variables)

        pybamm.citations.register("Timms2021")

    def get_fundamental_variables(self):
        # Get necessary parameters
        param = self.param
        L_cn = param.n.L_cc
        L_cp = param.p.L_cc
        sigma_cn = param.n.sigma_cc
        sigma_cp = param.p.sigma_cc

        # Set model variables: Note: we solve using a scaled version that is
        # better conditioned
        R_cn_scaled = pybamm.Variable(
            "Scaled negative current collector resistance [Ohm]",
            domain="current collector",
        )
        R_cp_scaled = pybamm.Variable(
            "Scaled positive current collector resistance [Ohm]",
            domain="current collector",
        )
        R_cn = R_cn_scaled / (L_cn * sigma_cn)
        R_cp = R_cp_scaled / (L_cp * sigma_cp)

        # Define effective current collector resistance
        if self.options["dimensionality"] == 1:
            R_cc_n = -pybamm.z_average(R_cn)
            R_cc_p = -pybamm.z_average(R_cp)
        elif self.options["dimensionality"] == 2:
            R_cc_n = -pybamm.yz_average(R_cn)
            R_cc_p = -pybamm.yz_average(R_cp)
        R_cc = R_cc_n + R_cc_p

        variables = {
            "Scaled negative current collector resistance [Ohm]": R_cn_scaled,
            "Negative current collector resistance [Ohm]": R_cn,
            "Scaled positive current collector resistance [Ohm]": R_cp_scaled,
            "Positive current collector resistance [Ohm]": R_cp,
            "Effective current collector resistance [Ohm]": R_cc,
            "Effective negative current collector resistance [Ohm]": R_cc_n,
            "Effective positive current collector resistance [Ohm]": R_cc_p,
        }

        # Add spatial variables
        var = pybamm.standard_spatial_vars
        if self.options["dimensionality"] == 1:
            variables.update({"z [m]": var.z})
        elif self.options["dimensionality"] == 2:
            variables.update({"y [m]": var.y, "z [m]": var.z})

        return variables

    def set_algebraic(self, variables):
        R_cn_scaled = variables["Scaled negative current collector resistance [Ohm]"]
        R_cp_scaled = variables["Scaled positive current collector resistance [Ohm]"]
        self.algebraic = {
            R_cn_scaled: pybamm.laplacian(R_cn_scaled) - pybamm.source(1, R_cn_scaled),
            R_cp_scaled: pybamm.laplacian(R_cp_scaled) - pybamm.source(1, R_cp_scaled),
        }

    def set_boundary_conditions(self, variables):
        R_cn_scaled = variables["Scaled negative current collector resistance [Ohm]"]
        R_cp_scaled = variables["Scaled positive current collector resistance [Ohm]"]

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
        R_cn_scaled = variables["Scaled negative current collector resistance [Ohm]"]
        R_cp_scaled = variables["Scaled positive current collector resistance [Ohm]"]
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
        # Process resistances
        R_cn = solution["Negative current collector resistance [Ohm]"]
        R_cp = solution["Positive current collector resistance [Ohm]"]
        R_cc = solution["Effective current collector resistance [Ohm]"]

        # Create callable combination of ProcessedVariable objects for potentials
        # and voltage
        def V(t):
            "Account for effective current collector resistance"
            return V_av(t) - I_av(t) * R_cc(t)

        def phi_s_cn(t, z, y=None):
            return R_cn(y=y, z=z) * I_av(t=t)

        def phi_s_cp(t, z, y=None):
            return V(t) - R_cp(y=y, z=z) * I_av(t=t)

        def V_cc(t, z, y=None):
            return phi_s_cp(t, z, y=y) - phi_s_cn(t, z, y=y)

        processed_vars = {
            "Negative current collector potential [V]": phi_s_cn,
            "Positive current collector potential [V]": phi_s_cp,
            "Local current collector potential difference [V]": V_cc,
            "Voltage [V]": V,
        }
        return processed_vars


class AlternativeEffectiveResistance2D(BaseEffectiveResistance):
    """
    A model which calculates the effective Ohmic resistance of the 2D current
    collectors in the limit of large electrical conductivity. This model assumes
    a uniform *current density* at the tabs and the solution is computed by first
    solving and auxilliary problem which is the related to the resistances.
    """

    def __init__(self):
        super().__init__()
        self.options = {"dimensionality": 2}
        self.name = "Effective resistance in current collector model (2D)"
        self.param = pybamm.LithiumIonParameters()

        # Get necessary parameters
        param = self.param
        L_cn = param.n.L_cc
        L_cp = param.p.L_cc
        L_tab_p = param.p.L_tab
        A_tab_p = L_cp * L_tab_p
        sigma_cn = param.n.sigma_cc
        sigma_cp = param.p.sigma_cc

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
            f_p: pybamm.laplacian(f_p)
            - pybamm.source(1, f_p)
            + c * pybamm.DefiniteIntegralVector(f_p, vector_type="column"),
            c: pybamm.yz_average(f_p),
        }

        # Boundary conditons
        pos_tab_bc = L_cp / A_tab_p
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
        R_cc_n = pybamm.yz_average(f_n) / (L_cn * sigma_cn)
        R_cc_p = pybamm.BoundaryIntegral(f_p, "positive tab") / (L_cp * sigma_cp)
        R_cc = R_cc_n + R_cc_p

        self.variables = {
            "Unit solution in negative current collector": f_n,
            "Unit solution in positive current collector": f_p,
            "Effective current collector resistance [Ohm]": R_cc,
            "Effective negative current collector resistance [Ohm]": R_cc_n,
            "Effective positive current collector resistance [Ohm]": R_cc_p,
        }

        # Add spatial variables
        var = pybamm.standard_spatial_vars
        self.variables.update({"y [m]": var.y, "z [m]": var.z})

        pybamm.citations.register("Timms2021")

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
        L_cn = param_values.evaluate(param.n.L_cc)
        L_cp = param_values.evaluate(param.p.L_cc)
        sigma_cn = param_values.evaluate(param.n.sigma_cc)
        sigma_cp = param_values.evaluate(param.p.sigma_cc)

        # Process unit solutions
        f_n = solution["Unit solution in negative current collector"]
        f_p = solution["Unit solution in positive current collector"]

        # Get effective resistance
        R_cc = solution["Effective current collector resistance [Ohm]"]

        # Create callable combination of ProcessedVariable objects for potentials
        def V(t):
            "Account for effective current collector resistance"
            return V_av(t) - I_av(t) * R_cc(t)

        def phi_s_cn(t, y, z):
            return -(I_av(t=t) / L_cn / sigma_cn) * f_n(y=y, z=z)

        def phi_s_cp(t, y, z):
            return V(t) + (I_av(t=t) / L_cp / sigma_cp) * f_p(y=y, z=z)

        def V_cc(t, y, z):
            return phi_s_cp(t, y, z) - phi_s_cn(t, y, z)

        processed_vars = {
            "Negative current collector potential [V]": phi_s_cn,
            "Positive current collector potential [V]": phi_s_cp,
            "Local current collector potential difference [V]": V_cc,
            "Voltage [V]": V,
        }
        return processed_vars
