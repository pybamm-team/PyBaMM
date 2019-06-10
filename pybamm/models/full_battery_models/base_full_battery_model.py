#
# Full battery base model class
#

import pybamm
import os


class BaseFullBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

    **Extends:** :class:`StandardBatteryBaseModel`
    """

    def __init__(self, options=None):
        super().__init__()
        self._extra_options = options
        self.set_standard_output_variables()

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
            "thermal": False,
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
                "Exchange-current density": None,
            }
        )

        self.variables.update(
            {
                "Negative electrode current density [A.m-2]": None,
                "Positive electrode current density [A.m-2]": None,
                "Electrolyte current density [A.m-2]": None,
                "Interfacial current density [A.m-2]": None,
                "Exchange-current density [A.m-2]": None,
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

