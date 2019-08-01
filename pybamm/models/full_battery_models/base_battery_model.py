#
# Base battery model class
#

import pybamm
import os
from collections import OrderedDict


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options
        self.set_standard_output_variables()
        self.submodels = OrderedDict()  # ordered dict not default in 3.5
        self._built = False

    @property
    def default_parameter_values(self):
        # Default parameter values, geometry, submesh, spatial methods and solver
        # Lion parameters left as default parameter set for tests
        input_path = os.path.join(
            pybamm.root_dir(), "input", "parameters", "lithium-ion"
        )
        return pybamm.ParameterValues(
            os.path.join(
                input_path, "mcmb2528_lif6-in-ecdmc_lico2_parameters_Dualfoil.csv"
            ),
            {
                "Typical current [A]": 1,
                "Current function": pybamm.GetConstantCurrent(
                    pybamm.standard_parameters_lithium_ion.I_typ
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
                "Negative electrode reaction rate": os.path.join(
                    input_path, "graphite_electrolyte_reaction_rate.py"
                ),
                "Positive electrode reaction rate": os.path.join(
                    input_path, "lico2_electrolyte_reaction_rate.py"
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
        if self.options["dimensionality"] == 0:
            return pybamm.Geometry("1D macro", "1+1D micro")
        elif self.options["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro", "1+1D micro")
        elif self.options["dimensionality"] == 2:
            return pybamm.Geometry("2+1D macro", "1+1D micro")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {
            var.x_n: 40,
            var.x_s: 25,
            var.x_p: 35,
            var.r_n: 10,
            var.r_p: 10,
            var.y: 10,
            var.z: 10,
        }

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
        }
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.SubMesh0D
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.Uniform1DSubMesh
        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.Scikit2DSubMesh
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods["current collector"] = pybamm.ZeroDimensionalMethod
        if self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement
        return base_spatial_methods

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScipySolver()

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        default_options = {
            "dimensionality": 0,
            "surface form": False,
            "convection": False,
            "thermal": None,
            "first-order potential": "linear",
            "side reactions": [],
            "interfacial surface area": "constant",
            "current collector": "uniform",
        }
        options = default_options
        # any extra options overwrite the default options
        if extra_options is not None:
            for name, opt in extra_options.items():
                if name in default_options:
                    options[name] = opt
                else:
                    raise pybamm.OptionError("option {} not recognised".format(name))

        # Some standard checks to make sure options are compatible
        if (
            isinstance(self, (pybamm.lead_acid.LOQS, pybamm.lead_acid.Composite))
            and options["surface form"] is False
        ):
            if len(options["side reactions"]) > 0:
                raise pybamm.OptionError(
                    """
                    must use surface formulation to solve {!s} with side reactions
                    """.format(
                        self
                    )
                )
        if options["surface form"] not in [False, "differential", "algebraic"]:
            raise pybamm.OptionError(
                "surface form '{}' not recognised".format(options["surface form"])
            )
        if options["current collector"] not in [
            "uniform",
            "potential pair",
            "potential pair quite conductive",
            "single particle potential pair",
        ]:
            raise pybamm.OptionError(
                "current collector model '{}' not recognised".format(
                    options["current collector"]
                )
            )
        if options["dimensionality"] not in [0, 1, 2]:
            raise pybamm.OptionError(
                "Dimension of current collectors must be 0, 1, or 2, not {}".format(
                    options["dimensionality"]
                )
            )
        if options["thermal"] not in [None, "lumped", "full"]:
            raise pybamm.OptionError(
                "Unknown thermal model '{}'".format(options["thermal"])
            )

        self._options = options

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
                "X-averaged negative electrode open circuit potential": None,
                "X-averaged positive electrode open circuit potential": None,
                "X-averaged open circuit voltage": None,
                "Measured open circuit voltage": None,
                "Terminal voltage": None,
            }
        )
        self.variables.update(
            {
                "Negative electrode open circuit potential [V]": None,
                "Positive electrode open circuit potential [V]": None,
                "X-averaged negative electrode open circuit potential [V]": None,
                "X-averaged positive electrode open circuit potential [V]": None,
                "X-averaged open circuit voltage [V]": None,
                "Measured open circuit voltage [V]": None,
                "Terminal voltage [V]": None,
            }
        )

        # Overpotentials
        self.variables.update(
            {
                "Negative reaction overpotential": None,
                "Positive reaction overpotential": None,
                "X-averaged negative reaction overpotential": None,
                "X-averaged positive reaction overpotential": None,
                "X-averaged reaction overpotential": None,
                "X-averaged electrolyte overpotential": None,
                "X-averaged solid phase ohmic losses": None,
            }
        )
        self.variables.update(
            {
                "Negative reaction overpotential [V]": None,
                "Positive reaction overpotential [V]": None,
                "X-averaged negative reaction overpotential [V]": None,
                "X-averaged positive reaction overpotential [V]": None,
                "X-averaged reaction overpotential [V]": None,
                "X-averaged electrolyte overpotential [V]": None,
                "X-averaged solid phase ohmic losses [V]": None,
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
        i_cell = pybamm.electrical_parameters.current_with_time
        i_cell_dim = pybamm.electrical_parameters.dimensional_current_density_with_time
        I = pybamm.electrical_parameters.dimensional_current_with_time
        self.variables.update(
            {
                "Total current density": i_cell,
                "Total current density [A.m-2]": i_cell_dim,
                "Current [A]": I,
            }
        )

        # Time
        time_scale = pybamm.electrical_parameters.timescale
        self.variables.update(
            {
                "Time": pybamm.t,
                "Time [s]": pybamm.t * time_scale,
                "Time [min]": pybamm.t * time_scale / 60,
                "Time [h]": pybamm.t * time_scale / 3600,
                "Discharge capacity [A.h]": I * pybamm.t * time_scale / 3600,
            }
        )

        # Spatial
        var = pybamm.standard_spatial_vars
        L_x = pybamm.geometric_parameters.L_x
        L_y = pybamm.geometric_parameters.L_y
        L_z = pybamm.geometric_parameters.L_z
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
        if self.options["dimensionality"] == 1:
            self.variables.update({"y": var.y, "y [m]": var.y * L_y})
        elif self.options["dimensionality"] == 2:
            self.variables.update(
                {"y": var.y, "y [m]": var.y * L_y, "z": var.z, "z [m]": var.z * L_z}
            )

    def build_model(self):
        pybamm.logger.info("Building {}".format(self.name))

        # Get the fundamental variables
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting fundamental variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            self.variables.update(submodel.get_fundamental_variables())

        # Get coupled variables
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting coupled variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            self.variables.update(submodel.get_coupled_variables(self.variables))

        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Setting rhs for {} submodel ({})".format(submodel_name, self.name)
            )
            submodel.set_rhs(self.variables)
            pybamm.logger.debug(
                "Setting algebraic for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            submodel.set_algebraic(self.variables)
            pybamm.logger.debug(
                "Setting boundary conditions for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            submodel.set_boundary_conditions(self.variables)
            pybamm.logger.debug(
                "Setting initial conditions for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            submodel.set_initial_conditions(self.variables)
            submodel.set_events(self.variables)
            pybamm.logger.debug(
                "Updating {} submodel ({})".format(submodel_name, self.name)
            )
            self.update(submodel)

        pybamm.logger.debug("Setting voltage variables")
        self.set_voltage_variables()

        pybamm.logger.debug("Setting SoC variables")
        self.set_soc_variables()

        self._built = True

    def set_thermal_submodel(self):

        if self.options["thermal"] is None:
            thermal_submodel = pybamm.thermal.Isothermal(self.param)
        elif self.options["thermal"] == "full":
            thermal_submodel = pybamm.thermal.Full(self.param)
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped(self.param)

        self.submodels["thermal"] = thermal_submodel

    def set_current_collector_submodel(self):

        if self.options["current collector"] == "uniform":
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
        elif self.options["current collector"] == "single particle potential pair":
            submodel = pybamm.current_collector.SingleParticlePotentialPair(self.param)
        self.submodels["current collector"] = submodel

    def set_voltage_variables(self):

        ocp_n = self.variables["Negative electrode open circuit potential"]
        ocp_p = self.variables["Positive electrode open circuit potential"]
        ocp_n_av = self.variables[
            "X-averaged negative electrode open circuit potential"
        ]
        ocp_p_av = self.variables[
            "X-averaged positive electrode open circuit potential"
        ]

        ocp_n_dim = self.variables["Negative electrode open circuit potential [V]"]
        ocp_p_dim = self.variables["Positive electrode open circuit potential [V]"]
        ocp_n_av_dim = self.variables[
            "X-averaged negative electrode open circuit potential [V]"
        ]
        ocp_p_av_dim = self.variables[
            "X-averaged positive electrode open circuit potential [V]"
        ]

        ocp_n_left = pybamm.boundary_value(ocp_n, "left")
        ocp_n_left_dim = pybamm.boundary_value(ocp_n_dim, "left")
        ocp_p_right = pybamm.boundary_value(ocp_p, "right")
        ocp_p_right_dim = pybamm.boundary_value(ocp_p_dim, "right")

        ocv_av = ocp_p_av - ocp_n_av
        ocv_av_dim = ocp_p_av_dim - ocp_n_av_dim
        ocv = ocp_p_right - ocp_n_left
        ocv_dim = ocp_p_right_dim - ocp_n_left_dim

        # overpotentials
        eta_r_n_av = self.variables[
            "X-averaged negative electrode reaction overpotential"
        ]
        eta_r_n_av_dim = self.variables[
            "X-averaged negative electrode reaction overpotential [V]"
        ]
        eta_r_p_av = self.variables[
            "X-averaged positive electrode reaction overpotential"
        ]
        eta_r_p_av_dim = self.variables[
            "X-averaged positive electrode reaction overpotential [V]"
        ]

        delta_phi_s_n_av = self.variables["X-averaged negative electrode ohmic losses"]
        delta_phi_s_n_av_dim = self.variables[
            "X-averaged negative electrode ohmic losses [V]"
        ]
        delta_phi_s_p_av = self.variables["X-averaged positive electrode ohmic losses"]
        delta_phi_s_p_av_dim = self.variables[
            "X-averaged positive electrode ohmic losses [V]"
        ]

        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av
        delta_phi_s_av_dim = delta_phi_s_p_av_dim - delta_phi_s_n_av_dim

        eta_r_av = eta_r_p_av - eta_r_n_av
        eta_r_av_dim = eta_r_p_av_dim - eta_r_n_av_dim

        # terminal voltage
        phi_s_cn = self.variables["Negative current collector potential"]
        phi_s_cp = self.variables["Positive current collector potential"]
        phi_s_cn_dim = self.variables["Negative current collector potential [V]"]
        phi_s_cp_dim = self.variables["Positive current collector potential [V]"]
        if self.options["dimensionality"] == 0:
            V = phi_s_cp
            V_dim = phi_s_cp_dim
        elif self.options["dimensionality"] == 1:
            # In 1D both tabs are at "right"
            V = pybamm.BoundaryValue(phi_s_cp, "right")
            V_dim = pybamm.BoundaryValue(phi_s_cp_dim, "right")
        elif self.options["dimensionality"] == 2:
            # In 2D left corresponds to the negative tab and right the positive tab
            V = pybamm.BoundaryValue(phi_s_cp, "right") - pybamm.BoundaryValue(
                phi_s_cn, "left"
            )
            V_dim = pybamm.BoundaryValue(phi_s_cp_dim, "right") - pybamm.BoundaryValue(
                phi_s_cn_dim, "left"
            )
        else:
            raise pybamm.ModelError(
                "Dimension of current collectors must be 0, 1, or 2, not {}".format(
                    self.options["dimensionality"]
                )
            )

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "X-averaged open circuit voltage": ocv_av,
                "Measured open circuit voltage": ocv,
                "X-averaged open circuit voltage [V]": ocv_av_dim,
                "Measured open circuit voltage [V]": ocv_dim,
                "X-averaged reaction overpotential": eta_r_av,
                "X-averaged reaction overpotential [V]": eta_r_av_dim,
                "X-averaged solid phase ohmic losses": delta_phi_s_av,
                "X-averaged solid phase ohmic losses [V]": delta_phi_s_av_dim,
                "Terminal voltage": V,
                "Terminal voltage [V]": V_dim,
            }
        )

        # Battery-wide variables
        eta_e_av_dim = self.variables.get("X-averaged electrolyte ohmic losses [V]", 0)
        eta_c_av_dim = self.variables.get(
            "X-averaged concentration overpotential [V]", 0
        )
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

        self.variables.update(
            {
                "X-averaged battery open circuit voltage [V]": ocv_av_dim * num_cells,
                "Measured battery open circuit voltage [V]": ocv_dim * num_cells,
                "X-averaged battery reaction overpotential [V]": eta_r_av_dim
                * num_cells,
                "X-averaged battery solid phase ohmic losses [V]": delta_phi_s_av_dim
                * num_cells,
                "X-averaged battery electrolyte ohmic losses [V]": eta_e_av_dim
                * num_cells,
                "X-averaged battery concentration overpotential [V]": eta_c_av_dim
                * num_cells,
                "Battery voltage [V]": V_dim * num_cells,
            }
        )

        # Cut-off voltage
        voltage = self.variables["Terminal voltage"]
        self.events["Minimum voltage"] = voltage - self.param.voltage_low_cut
        self.events["Maximum voltage"] = voltage - self.param.voltage_high_cut

    def set_soc_variables(self):
        """
        Set variables relating to the state of charge.
        This function is overriden by the base battery models
        """
        pass

    def process_parameters_and_discretise(self, symbol):
        """
        Process parameters and discretise a symbol using default parameter values,
        geometry, etc. Note that the model needs to be built first for this to be
        possible.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be processed

        Returns
        -------
        :class:`pybamm.Symbol`
            Processed symbol
        """
        if not self._built:
            self.build_model()

        # Set up parameters
        geometry = self.default_geometry
        parameter_values = self.default_parameter_values
        parameter_values.process_geometry(geometry)

        # Set up discretisation
        mesh = pybamm.Mesh(geometry, self.default_submesh_types, self.default_var_pts)
        disc = pybamm.Discretisation(mesh, self.default_spatial_methods)
        variables = list(self.rhs.keys()) + list(self.algebraic.keys())
        disc.set_variable_slices(variables)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol
