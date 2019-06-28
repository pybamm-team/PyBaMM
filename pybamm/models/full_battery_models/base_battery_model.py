#
# Base battery model class
#

import pybamm
import os


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None):
        super().__init__()
        self._extra_options = options
        self.set_standard_output_variables()
        self.submodels = {}

    @property
    def default_parameter_values(self):
        # Default parameter values, geometry, submesh, spatial methods and solver
        # Lion parameters left as default parameter set for tests
        input_path = os.path.join(os.getcwd(), "input", "parameters", "lithium-ion")
        return pybamm.ParameterValues(
            os.path.join(
                input_path, "mcmb2528_lif6-in-ecdmc_lico2_parameters_Dualfoil.csv"
            ),
            {
                "Typical current [A]": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
                "Electrolyte diffusivity": os.path.join(
                    input_path, "electrolyte_diffusivity_Capiglia1999.py"
                ),
                "Electrolyte conductivity": os.path.join(
                    input_path, "electrolyte_conductivity_Capiglia1999.py"
                ),
                "Negative electrode OCV": os.path.join(
                    input_path, "graphite_mcmb2528_ocp_Dualfoil.py"
                ),
                "Positive electrode OCV": os.path.join(
                    input_path, "lico2_ocp_Dualfoil.py"
                ),
                "Negative electrode diffusivity": os.path.join(
                    input_path, "graphite_mcmb2528_diffusivity_Dualfoil.py"
                ),
                "Positive electrode diffusivity": os.path.join(
                    input_path, "lico2_diffusivity_Dualfoil.py"
                ),
                "Negative electrode OCV entropic change": os.path.join(
                    input_path, "graphite_entropic_change_Moura.py"
                ),
                "Positive electrode OCV entropic change": os.path.join(
                    input_path, "lico2_entropic_change_Moura.py"
                ),
            },
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("1D macro", "1+1D micro")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {
            var.x_n: 40,
            var.x_s: 25,
            var.x_p: 35,
            var.r_n: 10,
            var.r_p: 10,
            var.z: 10,
        }

    @property
    def default_submesh_types(self):
        return {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }

    @property
    def default_spatial_methods(self):
        return {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        try:
            default_solver = pybamm.ScikitsOdeSolver()
        except ImportError:
            default_solver = pybamm.ScipySolver()

        return default_solver

    @property
    def options(self):
        default_options = {
            "bc_options": {"dimensionality": 0},
            "capacitance": False,
            "convection": False,
            "thermal": None,
            "first-order potential": "linear",
            "side reactions": [],
            "interfacial surface area": "constant",
            "Voltage": "On",
        }
        if self._extra_options is None:
            options = default_options
        else:
            # any extra options overwrite the default options
            options = {**default_options, **self._extra_options}

        # Some standard checks to make sure options are compatible
        if (
            isinstance(self, (pybamm.lead_acid.LOQS, pybamm.lead_acid.Composite))
            and options["capacitance"] is False
            and options["bc_options"]["dimensionality"] == 1
        ):
            raise pybamm.ModelError(
                "must use capacitance formulation to solve {!s} in 2D".format(self)
            )

            if len(options["side reactions"]) > 0:
                raise pybamm.ModelError(
                    """
                    must use capacitance formulation to solve {!s} with side reactions
                    """.format(
                        self
                    )
                )

        return options

    def set_standard_output_variables(self):
        # Standard output variables
        # Interfacial current
        self.variables.update(
            {
                "Negative electrode current density": None,
                "Positive electrode current density": None,
                "Electrolyte current density": None,
                "Interfacial current density": None,
                "Exchange current density": None,
            }
        )

        self.variables.update(
            {
                "Negative electrode current density [A.m-2]": None,
                "Positive electrode current density [A.m-2]": None,
                "Electrolyte current density [A.m-2]": None,
                "Interfacial current density [A.m-2]": None,
                "Exchange current density [A.m-2]": None,
            }
        )
        # Voltage
        self.variables.update(
            {
                "Negative electrode open circuit potential": None,
                "Positive electrode open circuit potential": None,
                "Average negative electrode open circuit potential": None,
                "Average positive electrode open circuit potential": None,
                "Average open circuit voltage": None,
                "Measured open circuit voltage": None,
                "Terminal voltage": None,
            }
        )

        self.variables.update(
            {
                "Negative electrode open circuit potential [V]": None,
                "Positive electrode open circuit potential [V]": None,
                "Average negative electrode open circuit potential [V]": None,
                "Average positive electrode open circuit potential [V]": None,
                "Average open circuit voltage [V]": None,
                "Measured open circuit voltage [V]": None,
                "Terminal voltage [V]": None,
            }
        )

        # Overpotentials
        self.variables.update(
            {
                "Negative reaction overpotential": None,
                "Positive reaction overpotential": None,
                "Average negative reaction overpotential": None,
                "Average positive reaction overpotential": None,
                "Average reaction overpotential": None,
                "Average electrolyte overpotential": None,
                "Average solid phase ohmic losses": None,
            }
        )

        self.variables.update(
            {
                "Negative reaction overpotential [V]": None,
                "Positive reaction overpotential [V]": None,
                "Average negative reaction overpotential [V]": None,
                "Average positive reaction overpotential [V]": None,
                "Average reaction overpotential [V]": None,
                "Average electrolyte overpotential [V]": None,
                "Average solid phase ohmic losses [V]": None,
            }
        )
        # Concentration
        self.variables.update(
            {
                "Electrolyte concentration": None,
                "Electrolyte concentration [mol.m-3]": None,
            }
        )

        # Potential
        self.variables.update(
            {
                "Negative electrode potential": None,
                "Positive electrode potential": None,
                "Electrolyte potential": None,
            }
        )

        self.variables = {}

        # Current
        icell = pybamm.electrical_parameters.current_with_time
        icell_dim = pybamm.electrical_parameters.dimensional_current_density_with_time
        I = pybamm.electrical_parameters.dimensional_current_with_time
        self.variables.update(
            {
                "Total current density": icell,
                "Total current density [A.m-2]": icell_dim,
                "Current [A]": I,
            }
        )
        # Time
        self.variables.update({"Time": pybamm.t})
        # x-position
        var = pybamm.standard_spatial_vars
        L_x = pybamm.geometric_parameters.L_x
        self.variables.update(
            {
                "x": var.x,
                "x [m]": var.x * L_x,
                "x_n": var.x_n,
                "x_n [m]": var.x_n * L_x,
                "x_s": var.x_s,
                "x_s [m]": var.x_s * L_x,
                "x_p": var.x_p,
                "x_p [m]": var.x_p * L_x,
            }
        )

    def build_model(self):

        # Get the fundamental variables
        for submodel in self.submodels.values():
            self.variables.update(submodel.get_fundamental_variables())

        # Get coupled variables
        for submodel in self.submodels.values():
            self.variables.update(submodel.get_coupled_variables(self.variables))

        # Set model equations
        for submodel in self.submodels.values():
            submodel.set_rhs(self.variables)
            submodel.set_algebraic(self.variables)
            submodel.set_boundary_conditions(self.variables)
            submodel.set_initial_conditions(self.variables)
            submodel.set_events(self.variables)
            self.update(submodel)

        # This if statement only exists for the reaction diffusion model.
        if self.options["Voltage"] != "Off":
            self.set_voltage_variables()

    def set_thermal_submodel(self):

        if self.options["thermal"] is None:
            thermal_submodel = pybamm.thermal.Isothermal(self.param)
        elif self.options["thermal"] == "full":
            thermal_submodel = pybamm.thermal.FullModel(self.param)
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.LumpedModel(self.param)
        else:
            raise KeyError("Unknown type of thermal model")

        self.submodels["thermal"] = thermal_submodel

    def set_voltage_variables(self):

        ocp_n = self.variables["Negative electrode open circuit potential"]
        ocp_p = self.variables["Positive electrode open circuit potential"]
        ocp_n_av = self.variables["Average negative electrode open circuit potential"]
        ocp_p_av = self.variables["Average positive electrode open circuit potential"]

        ocp_n_dim = self.variables["Negative electrode open circuit potential [V]"]
        ocp_p_dim = self.variables["Positive electrode open circuit potential [V]"]
        ocp_n_av_dim = self.variables[
            "Average negative electrode open circuit potential [V]"
        ]
        ocp_p_av_dim = self.variables[
            "Average positive electrode open circuit potential [V]"
        ]

        ocp_n_left = pybamm.BoundaryValue(ocp_n, "left")
        ocp_n_left_dim = pybamm.BoundaryValue(ocp_n_dim, "left")
        ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
        ocp_p_right_dim = pybamm.BoundaryValue(ocp_p_dim, "right")

        ocv_av = ocp_p_av - ocp_n_av
        ocv_av_dim = ocp_p_av_dim - ocp_n_av_dim
        ocv = ocp_p_right - ocp_n_left
        ocv_dim = ocp_p_right_dim - ocp_n_left_dim

        # overpotentials
        eta_r_n_av = self.variables["Average negative electrode reaction overpotential"]
        eta_r_n_av_dim = self.variables[
            "Average negative electrode reaction overpotential [V]"
        ]
        eta_r_p_av = self.variables["Average positive electrode reaction overpotential"]
        eta_r_p_av_dim = self.variables[
            "Average positive electrode reaction overpotential [V]"
        ]

        delta_phi_s_n_av = self.variables["Average negative electrode ohmic losses"]
        delta_phi_s_n_av_dim = self.variables[
            "Average negative electrode ohmic losses [V]"
        ]
        delta_phi_s_p_av = self.variables["Average positive electrode ohmic losses"]
        delta_phi_s_p_av_dim = self.variables[
            "Average positive electrode ohmic losses [V]"
        ]

        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av
        delta_phi_s_av_dim = delta_phi_s_p_av_dim - delta_phi_s_n_av_dim

        eta_r_av = eta_r_p_av - eta_r_n_av
        eta_r_av_dim = eta_r_p_av_dim - eta_r_n_av_dim

        # terminal voltage
        phi_s_p = self.variables["Positive electrode potential"]
        phi_s_p_dim = self.variables["Positive electrode potential [V]"]
        V = pybamm.BoundaryValue(phi_s_p, "right")
        V_dim = pybamm.BoundaryValue(phi_s_p_dim, "right")

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "Average open circuit voltage": ocv_av,
                "Measured open circuit voltage": ocv,
                "Average open circuit voltage [V]": ocv_av_dim,
                "Measured open circuit voltage [V]": ocv_dim,
                "Average reaction overpotential": eta_r_av,
                "Average reaction overpotential [V]": eta_r_av_dim,
                "Average solid phase ohmic losses": delta_phi_s_av,
                "Average solid phase ohmic losses [V]": delta_phi_s_av_dim,
                "Terminal voltage": V,
                "Terminal voltage [V]": V_dim,
            }
        )

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events["Minimum voltage"] = voltage - self.param.voltage_low_cut

