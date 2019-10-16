#
# Base battery model class
#

import pybamm
import os


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

    Attributes
    ----------

    options: dict
        A dictionary of options to be passed to the model. The options that can
        be set are listed below. Note that not all of the options are compatible with
        each other and with all of the models implemented in PyBaMM.

            * "dimensionality" : int, optional
                Sets the dimension of the current collector problem. Can be 0
                (default), 1 or 2.
            * "surface form" : bool or str, optional
                Whether to use the surface formulation of the problem. Can be False
                (default), "differential" or "algebraic". Must be 'False' for
                lithium-ion models.
            * "convection" : bool or str, optional
                Whether to include the effects of convection in the model. Can be
                False (default), "differential" or "algebraic". Must be 'False' for
                lithium-ion models.
            * "side reactions" : list, optional
                Contains a list of any side reactions to include. Default is []. If this
                list is not empty (i.e. side reactions are included in the model), then
                "surface form" cannot be 'False'.
            * "interfacial surface area" : str, optional
                Sets the model for the interfacial surface area. Can be "constant"
                (default) or "varying". Not currently implemented in any of the models.
            * "current collector" : str, optional
                Sets the current collector model to use. Can be "uniform" (default),
                "potential pair", "potential pair quite conductive", "single particle
                potential pair" or "set external potential". The submodel
                "single particle potential pair" can only be used with lithium-ion
                single particle models. The submodel "set external potential" can only
                be used with the SPM.
            * "particle" : str, optional
                Sets the submodel to use to describe behaviour within the particle.
                Can be "Fickian diffusion" (default) or "fast diffusion".
            * "thermal" : str, optional
                Sets the thermal model to use. Can be "isothermal" (default),
                "x-full", "x-lumped", "xyz-lumped", "lumped" or "set external
                temperature". Must be "isothermal" for lead-acid models. If the
                option "set external temperature" is selected then "dimensionality"
                must be 1.
            * "thermal current collector" : bool, optional
                Whether to include thermal effects in the current collector in
                one-dimensional models (default is False). Note that this option
                only takes effect if "dimensionality" is 0. If "dimensionality"
                is 1 or 2 current collector effects are always included. Must be 'False'
                for lead-acid models.


    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options
        self.set_standard_output_variables()
        self.submodels = {}
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
                "Typical timescale [s]": 1,
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
            base_submeshes["current collector"] = pybamm.ScikitUniform2DSubMesh
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
        elif self.options["dimensionality"] == 1:
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
            "side reactions": [],
            "interfacial surface area": "constant",
            "current collector": "uniform",
            "particle": "Fickian diffusion",
            "thermal": "isothermal",
            "thermal current collector": False,
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
            "set external potential",
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
        if options["thermal"] not in [
            "isothermal",
            "x-full",
            "x-lumped",
            "xyz-lumped",
            "lumped",
            "set external temperature",
        ]:
            raise pybamm.OptionError(
                "Unknown thermal model '{}'".format(options["thermal"])
            )
        if options["particle"] not in ["Fickian diffusion", "fast diffusion"]:
            raise pybamm.OptionError(
                "particle model '{}' not recognised".format(options["particle"])
            )

        # Options that are incompatible with models
        if isinstance(self, pybamm.lithium_ion.BaseModel):
            if options["surface form"] is not False:
                raise pybamm.OptionError(
                    "surface form not implemented for lithium-ion models"
                )
            if options["convection"] is True:
                raise pybamm.OptionError(
                    "convection not implemented for lithium-ion models"
                )
        if isinstance(self, pybamm.lead_acid.BaseModel):
            if options["thermal"] != "isothermal":
                raise pybamm.OptionError(
                    "thermal effects not implemented for lead-acid models"
                )
            if options["thermal current collector"] is True:
                raise pybamm.OptionError(
                    "thermal effects not implemented for lead-acid models"
                )
        if options[
            "current collector"
        ] == "single particle potenetial pair" and not isinstance(
            self, (pybamm.lithium_ion.SPM, pybamm.lithium_ion.SPMe)
        ):
            raise pybamm.OptionError(
                "option {} only compatible with SPM or SPMe".format(
                    options["current collector"]
                )
            )
        if options["current collector"] == "set external potential" and not isinstance(
            self, pybamm.lithium_ion.SPM
        ):
            raise pybamm.OptionError(
                "option {} only compatible with SPM".format(
                    options["current collector"]
                )
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

        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal(self.param)

        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                if self.options["thermal current collector"] is False:
                    thermal_submodel = pybamm.thermal.x_lumped.NoCurrentCollector(
                        self.param
                    )
                elif self.options["thermal current collector"] is True:
                    thermal_submodel = pybamm.thermal.x_lumped.CurrentCollector0D(
                        self.param
                    )
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.x_lumped.CurrentCollector1D(
                    self.param
                )
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.x_lumped.CurrentCollector2D(
                    self.param
                )

        elif self.options["thermal"] == "x-full":
            if self.options["dimensionality"] == 0:
                if self.options["thermal current collector"] is False:
                    thermal_submodel = pybamm.thermal.x_full.NoCurrentCollector(
                        self.param
                    )
                elif self.options["thermal current collector"] is True:
                    raise NotImplementedError(
                        """X-full thermal submodels do
                    not yet account for current collector"""
                    )
            elif self.options["dimensionality"] == 1:
                raise NotImplementedError(
                    """X-full thermal submodels do not
                yet support 1D current collectors"""
                )
            elif self.options["dimensionality"] == 2:
                raise NotImplementedError(
                    """X-full thermal submodels do
                    not yet support 2D current collectors"""
                )

        elif self.options["thermal"] == "xyz-lumped":
            if self.options["dimensionality"] == 0:
                # note here we will just call the x_lumped model
                # because it is equivalent
                if self.options["thermal current collector"] is False:
                    thermal_submodel = pybamm.thermal.x_lumped.NoCurrentCollector(
                        self.param
                    )
                elif self.options["thermal current collector"] is True:
                    thermal_submodel = pybamm.thermal.x_lumped.CurrentCollector0D(
                        self.param
                    )
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.xyz_lumped.CurrentCollector1D(
                    self.param
                )
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.xyz_lumped.CurrentCollector2D(
                    self.param
                )

        elif self.options["thermal"] == "lumped":
            # Easy option for returning a single Temperature regardless of choice of
            # current collector model. Note: Always includes current collector effects
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.x_lumped.CurrentCollector0D(
                    self.param
                )
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.xyz_lumped.CurrentCollector1D(
                    self.param
                )
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.xyz_lumped.CurrentCollector2D(
                    self.param
                )

        elif self.options["thermal"] == "set external temperature":
            if self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.x_lumped.SetTemperature1D(self.param)
            elif self.options["dimensionality"] in [0, 2]:
                raise NotImplementedError(
                    """Set temperature model only implemented for 1D current
                    collectors"""
                )
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
        elif self.options["current collector"] == "set external potential":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.SetPotentialSingleParticle1plus1D(
                    self.param
                )
            elif self.options["dimensionality"] in [0, 2]:
                raise NotImplementedError(
                    """Set potential model only implemented for 1D current
                    collectors"""
                )
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

        # terminal voltage (Note: phi_s_cn is zero at the negative tab)
        phi_s_cp = self.variables["Positive current collector potential"]
        phi_s_cp_dim = self.variables["Positive current collector potential [V]"]
        if self.options["dimensionality"] == 0:
            V = phi_s_cp
            V_dim = phi_s_cp_dim
        elif self.options["dimensionality"] in [1, 2]:
            V = pybamm.BoundaryValue(phi_s_cp, "positive tab")
            V_dim = pybamm.BoundaryValue(phi_s_cp_dim, "positive tab")

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

    def process_parameters_and_discretise(self, symbol, parameter_values, disc):
        """
        Process parameters and discretise a symbol using supplied parameter values
        and discretisation. Note: care should be taken if using spatial operators
        on dimensional symbols. Operators in pybamm are written in non-dimensional
        form, so may need to be scaled by the appropriate length scale. It is
        recommended to use this method on non-dimensional symbols.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol to be processed
        parameter_values : :class:`pybamm.ParameterValues`
            The parameter values to use during processing
        disc : :class:`pybamm.Discretisation`
            The discrisation to use

        Returns
        -------
        :class:`pybamm.Symbol`
            Processed symbol
        """
        # Set y slices
        if disc.y_slices == {}:
            variables = list(self.rhs.keys()) + list(self.algebraic.keys())
            disc.set_variable_slices(variables)

        # Set boundary condtions
        if disc.bcs == {}:
            disc.bcs = disc.process_boundary_conditions(self)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol
