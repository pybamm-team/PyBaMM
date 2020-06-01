#
# Base battery model class
#

import pybamm
import warnings


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
                (default), "differential" or "algebraic".
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
                "potential pair" or "potential pair quite conductive".
            * "particle" : str, optional
                Sets the submodel to use to describe behaviour within the particle.
                Can be "Fickian diffusion" (default) or "fast diffusion".
            * "thermal" : str, optional
                Sets the thermal model to use. Can be "isothermal" (default), "lumped",
                "x-lumped", or "x-full".
            * "external submodels" : list
                A list of the submodels that you would like to supply an external
                variable for instead of solving in PyBaMM. The entries of the lists
                are strings that correspond to the submodel names in the keys
                of `self.submodels`.
            * "sei" : str
                Set the sei submodel to be used. Options are:

                - None: :class:`pybamm.sei.NoSEI` (no SEI growth)
                - "constant": :class:`pybamm.sei.Constant` (constant SEI thickness)
                - "reaction limited": :class:`pybamm.sei.ReactionLimited`
                - "solvent-diffusion limited": \
                    :class:`pybamm.sei.SolventDiffusionLimited`
                - "electron-migration limited": \
                    :class:`pybamm.sei.ElectronMigrationLimited`
                - "interstitial-diffusion limited": \
                    :class:`pybamm.sei.InterstitialDiffusionLimited`
            * "sei film resistance" : str
                Set the submodel for additional term in the overpotential due to SEI.
                The default value is "None" if the "sei" option is "None", and
                "distributed" otherwise. This is because the "distributed" model is more
                complex than the model with no additional resistance, which adds
                unnecessary complexity if there is no SEI in the first place

                - None: no additional resistance\

                    .. math::
                        \\eta_r = \\frac{F}{RT} * (\\phi_s - \\phi_e - U)

                - "distributed": properly included additional resistance term\

                    .. math::
                        \\eta_r = \\frac{F}{RT}
                        * (\\phi_s - \\phi_e - U - R_{sei} * L_{sei} * j)

                - "average": constant additional resistance term (approximation to the \
                    true model). This model can give similar results to the \
                    "distributed" case without needing to make j an algebraic state\

                    .. math::
                        \\eta_r = \\frac{F}{RT}
                        * (\\phi_s - \\phi_e - U - R_{sei} * L_{sei} * \\frac{I}{aL})


    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options
        self.submodels = {}
        self._built = False
        self._built_fundamental_and_external = False

    @property
    def default_parameter_values(self):
        # Default parameter values
        # Lion parameters left as default parameter set for tests
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        return pybamm.battery_geometry(
            current_collector_dimension=self.options["dimensionality"]
        )

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {
            var.x_n: 20,
            var.x_s: 20,
            var.x_p: 20,
            var.r_n: 10,
            var.r_p: 10,
            var.y: 10,
            var.z: 10,
        }

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.MeshGenerator(pybamm.SubMesh0D)
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.MeshGenerator(
                pybamm.Uniform1DSubMesh
            )
        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.MeshGenerator(
                pybamm.ScikitUniform2DSubMesh
            )
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
        }
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods[
                "current collector"
            ] = pybamm.ZeroDimensionalSpatialMethod()
        elif self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume()
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement()
        return base_spatial_methods

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        default_options = {
            "operating mode": "current",
            "dimensionality": 0,
            "surface form": False,
            "convection": False,
            "side reactions": [],
            "interfacial surface area": "constant",
            "current collector": "uniform",
            "particle": "Fickian diffusion",
            "thermal": "isothermal",
            "cell_geometry": None,
            "external submodels": [],
            "sei": None,
        }
        # Change the default for cell geometry based on which thermal option is provided
        extra_options = extra_options or {}
        thermal_option = extra_options.get(
            "thermal", None
        )  # return None if option not given
        if thermal_option is None or thermal_option in ["isothermal", "lumped"]:
            default_options["cell_geometry"] = "arbitrary"
        else:
            default_options["cell_geometry"] = "pouch"
        # The "cell_geometry" option will still be overridden by extra_options if
        # provided

        # Change the default for SEI film resistance based on which sei option is
        # provided
        # extra_options = extra_options or {}
        sei_option = extra_options.get("sei", None)  # return None if option not given
        if sei_option is None:
            default_options["sei film resistance"] = None
        else:
            default_options["sei film resistance"] = "distributed"
        # The "sei film resistance" option will still be overridden by extra_options if
        # provided

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

        # Options that are incompatible with models
        if isinstance(self, pybamm.lithium_ion.BaseModel):
            if options["convection"] is not False:
                raise pybamm.OptionError(
                    "convection not implemented for lithium-ion models"
                )
            if (
                options["thermal"] in ["x-lumped", "x-full"]
                and options["cell_geometry"] != "pouch"
            ):
                raise pybamm.OptionError(
                    options["thermal"] + " model must have pouch geometry."
                )
        if isinstance(self, pybamm.lead_acid.BaseModel):
            if options["thermal"] != "isothermal" and options["dimensionality"] != 0:
                raise pybamm.OptionError(
                    "Lead-acid models can only have thermal "
                    "effects if dimensionality is 0."
                )
            if options["sei"] is not None or options["sei film resistance"] is not None:
                raise pybamm.OptionError("Lead-acid models cannot have SEI formation")

        # Some standard checks to make sure options are compatible
        if not (
            options["operating mode"] in ["current", "voltage", "power"]
            or callable(options["operating mode"])
        ):
            raise pybamm.OptionError(
                "operating mode '{}' not recognised".format(options["operating mode"])
            )
        if (
            isinstance(self, (pybamm.lead_acid.LOQS, pybamm.lead_acid.Composite))
            and options["surface form"] is False
        ):
            if len(options["side reactions"]) > 0:
                raise pybamm.OptionError(
                    """must use surface formulation to solve {!s} with side reactions
                    """.format(
                        self
                    )
                )
        if options["surface form"] not in [False, "differential", "algebraic"]:
            raise pybamm.OptionError(
                "surface form '{}' not recognised".format(options["surface form"])
            )
        if options["convection"] not in [
            False,
            "uniform transverse",
            "full transverse",
        ]:
            raise pybamm.OptionError(
                "convection option '{}' not recognised".format(options["convection"])
            )
        if options["current collector"] not in [
            "uniform",
            "potential pair",
            "potential pair quite conductive",
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
        if options["thermal"] not in ["isothermal", "lumped", "x-lumped", "x-full"]:
            raise pybamm.OptionError(
                "Unknown thermal model '{}'".format(options["thermal"])
            )
        if options["cell_geometry"] not in ["arbitrary", "pouch"]:
            raise pybamm.OptionError(
                "Unknown geometry '{}'".format(options["cell_geometry"])
            )
        if options["sei"] not in [
            None,
            "constant",
            "reaction limited",
            "solvent-diffusion limited",
            "electron-migration limited",
            "interstitial-diffusion limited",
        ]:
            raise pybamm.OptionError("Unknown sei model '{}'".format(options["sei"]))
        if options["sei film resistance"] not in [None, "distributed", "average"]:
            raise pybamm.OptionError(
                "Unknown sei film resistance model '{}'".format(
                    options["sei film resistance"]
                )
            )

        if options["dimensionality"] == 0:
            if options["current collector"] not in ["uniform"]:
                raise pybamm.OptionError(
                    "current collector model must be uniform in 0D model"
                )
            if options["convection"] == "full transverse":
                raise pybamm.OptionError(
                    "cannot have transverse convection in 0D model"
                )
        if options["particle"] not in ["Fickian diffusion", "fast diffusion"]:
            raise pybamm.OptionError(
                "particle model '{}' not recognised".format(options["particle"])
            )

        if options["thermal"] == "x-lumped" and options["dimensionality"] == 1:
            warnings.warn(
                "1+1D Thermal models are only valid if both tabs are "
                "placed at the top of the cell."
            )

        self._options = options

    def set_standard_output_variables(self):
        # Time
        self.variables.update(
            {
                "Time": pybamm.t,
                "Time [s]": pybamm.t * self.timescale,
                "Time [min]": pybamm.t * self.timescale / 60,
                "Time [h]": pybamm.t * self.timescale / 3600,
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
            self.variables.update({"z": var.z, "z [m]": var.z * L_z})
        elif self.options["dimensionality"] == 2:
            self.variables.update(
                {"y": var.y, "y [m]": var.y * L_y, "z": var.z, "z [m]": var.z * L_z}
            )

        # Initialize "total reaction" variables
        # These will get populated by the "get_coupled_variables" methods, and then used
        # later by "set_rhs" or "set_algebraic", which ensures that we always have
        # added all the necessary variables by the time the sum is used
        self.variables.update(
            {
                "Sum of electrolyte reaction source terms": 0,
                "Sum of negative electrode electrolyte reaction source terms": 0,
                "Sum of positive electrode electrolyte reaction source terms": 0,
                "Sum of x-averaged negative electrode "
                "electrolyte reaction source terms": 0,
                "Sum of x-averaged positive electrode "
                "electrolyte reaction source terms": 0,
                "Sum of interfacial current densities": 0,
                "Sum of negative electrode interfacial current densities": 0,
                "Sum of positive electrode interfacial current densities": 0,
                "Sum of x-averaged negative electrode interfacial current densities": 0,
                "Sum of x-averaged positive electrode interfacial current densities": 0,
            }
        )

    def build_fundamental_and_external(self):
        # Get the fundamental variables
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting fundamental variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            self.variables.update(submodel.get_fundamental_variables())

        # set the submodels that are external
        for sub in self.options["external submodels"]:
            self.submodels[sub].external = True

        # Set any external variables
        self.external_variables = []
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.debug(
                "Getting external variables for {} submodel ({})".format(
                    submodel_name, self.name
                )
            )
            external_variables = submodel.get_external_variables()

            self.external_variables += external_variables

        self._built_fundamental_and_external = True

    def build_coupled_variables(self):
        # Note: pybamm will try to get the coupled variables for the submodels in the
        # order they are set by the user. If this fails for a particular submodel,
        # return to it later and try again. If setting coupled variables fails and
        # there are no more submodels to try, raise an error.
        submodels = list(self.submodels.keys())
        count = 0
        # For this part the FuzzyDict of variables is briefly converted back into a
        # normal dictionary for speed with KeyErrors
        self._variables = dict(self._variables)
        while len(submodels) > 0:
            count += 1
            for submodel_name, submodel in self.submodels.items():
                if submodel_name in submodels:
                    pybamm.logger.debug(
                        "Getting coupled variables for {} submodel ({})".format(
                            submodel_name, self.name
                        )
                    )
                    try:
                        self.variables.update(
                            submodel.get_coupled_variables(self.variables)
                        )
                        submodels.remove(submodel_name)
                    except KeyError as key:
                        if len(submodels) == 1 or count == 100:
                            # no more submodels to try
                            raise pybamm.ModelError(
                                "Missing variable for submodel '{}': {}.\n".format(
                                    submodel_name, key
                                )
                                + "Check the selected "
                                "submodels provide all of the required variables."
                            )
                        else:
                            # try setting coupled variables on next loop through
                            pybamm.logger.debug(
                                "Can't find {}, trying other submodels first".format(
                                    key
                                )
                            )
        # Convert variables back into FuzzyDict
        self._variables = pybamm.FuzzyDict(self._variables)

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            if submodel.external is False:
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

    def build_model(self):

        # Check if already built
        if self._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info("Building {}".format(self.name))

        if self._built_fundamental_and_external is False:
            self.build_fundamental_and_external()

        self.build_coupled_variables()

        self.build_model_equations()

        pybamm.logger.debug("Setting voltage variables ({})".format(self.name))
        self.set_voltage_variables()

        pybamm.logger.debug("Setting SoC variables ({})".format(self.name))
        self.set_soc_variables()

        # Massive hack for consistent delta_phi = phi_s - phi_e with SPMe
        # This needs to be corrected
        if isinstance(self, pybamm.lithium_ion.SPMe):
            for domain in ["Negative", "Positive"]:
                phi_s = self.variables[domain + " electrode potential"]
                phi_e = self.variables[domain + " electrolyte potential"]
                delta_phi = phi_s - phi_e
                s = self.submodels[domain.lower() + " interface"]
                var = s._get_standard_surface_potential_difference_variables(delta_phi)
                self.variables.update(var)

        self._built = True

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            self.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(
                self.param
            )
        elif self.options["operating mode"] == "voltage":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.VoltageFunctionControl(self.param)
        elif self.options["operating mode"] == "power":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.PowerFunctionControl(self.param)
        elif callable(self.options["operating mode"]):
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"]
            )

    def set_tortuosity_submodels(self):
        self.submodels["electrolyte tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrolyte"
        )
        self.submodels["electrode tortuosity"] = pybamm.tortuosity.Bruggeman(
            self.param, "Electrode"
        )

    def set_thermal_submodel(self):

        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal(self.param)

        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped(
                self.param,
                self.options["dimensionality"],
                self.options["cell_geometry"],
            )

        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                # With 0D current collectors x-lumped is equivalent to lumped pouch
                thermal_submodel = pybamm.thermal.Lumped(self.param, geometry="pouch")
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector1D(
                    self.param
                )
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector2D(
                    self.param
                )

        elif self.options["thermal"] == "x-full":
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.OneDimensionalX(self.param)
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

        self.submodels["thermal"] = thermal_submodel

    def set_current_collector_submodel(self):

        if self.options["current collector"] in ["uniform"]:
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
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

        # SEI film overpotential
        eta_sei_n_av = self.variables[
            "X-averaged negative electrode sei film overpotential"
        ]
        eta_sei_p_av = self.variables[
            "X-averaged positive electrode sei film overpotential"
        ]
        eta_sei_n_av_dim = self.variables[
            "X-averaged negative electrode sei film overpotential [V]"
        ]
        eta_sei_p_av_dim = self.variables[
            "X-averaged positive electrode sei film overpotential [V]"
        ]
        eta_sei_av = eta_sei_n_av + eta_sei_p_av
        eta_sei_av_dim = eta_sei_n_av_dim + eta_sei_p_av_dim

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "X-averaged open circuit voltage": ocv_av,
                "Measured open circuit voltage": ocv,
                "X-averaged open circuit voltage [V]": ocv_av_dim,
                "Measured open circuit voltage [V]": ocv_dim,
                "X-averaged reaction overpotential": eta_r_av,
                "X-averaged reaction overpotential [V]": eta_r_av_dim,
                "X-averaged sei film overpotential": eta_sei_av,
                "X-averaged sei film overpotential [V]": eta_sei_av_dim,
                "X-averaged solid phase ohmic losses": delta_phi_s_av,
                "X-averaged solid phase ohmic losses [V]": delta_phi_s_av_dim,
            }
        )

        # Battery-wide variables
        V_dim = self.variables["Terminal voltage [V]"]
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
        self.events.append(
            pybamm.Event(
                "Minimum voltage",
                voltage - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage",
                voltage - self.param.voltage_high_cut,
                pybamm.EventType.TERMINATION,
            )
        )

        # Power
        I_dim = self.variables["Current [A]"]
        self.variables.update({"Terminal power [W]": I_dim * V_dim})

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

        # Set boundary condtions (also requires setting parameter values)
        if disc.bcs == {}:
            self.boundary_conditions = parameter_values.process_boundary_conditions(
                self
            )
            disc.bcs = disc.process_boundary_conditions(self)

        # Process
        param_symbol = parameter_values.process_symbol(symbol)
        disc_symbol = disc.process_symbol(param_symbol)

        return disc_symbol
