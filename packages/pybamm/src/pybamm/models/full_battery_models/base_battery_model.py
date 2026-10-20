#
# Base battery model class
#

import pybamm
from pybamm.expression_tree.operations.serialise import Serialise
from pybamm.models.full_battery_models.battery_model_options import (
    BatteryModelDomainOptions,
    BatteryModelOptions,
    BatteryModelPhaseOptions,
    LegacyOptions,
    _warn_legacy_defaults,
    iter_option_leaves,
    option_values_match,
    restore_saved_options,
)

# saved pickles refer to the option classes by this module's path
__all__ = [
    "BaseBatteryModel",
    "BatteryModelDomainOptions",
    "BatteryModelOptions",
    "BatteryModelPhaseOptions",
]


class BaseBatteryModel(pybamm.BaseModel):
    """
    Base model class with some default settings and required variables

    Parameters
    ----------
    options : dict-like, optional
        A dictionary of options to be passed to the model. If this is a dict (and not
        a subtype of dict), it will be processed by :class:`pybamm.BatteryModelOptions`
        to ensure that the options are valid. If this is a subtype of dict, it is
        assumed that the options have already been processed and are valid. This allows
        for the use of custom options classes. The default options are given by
        :class:`pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model. The default is "Unnamed battery model".
    """

    def __init__(self, options=None, name="Unnamed battery model"):
        super().__init__(name)
        self.options = options

    @classmethod
    def deserialise(cls, properties: dict):
        """
        Create a model instance from a serialised object.
        """

        # append the model name with _saved to differentiate
        instance = cls(
            options=restore_saved_options(
                properties["options"], properties.get("pybamm_version")
            ),
            name=properties["name"] + "_saved",
        )

        return cls.generic_deserialise(instance, properties)

    @property
    def default_geometry(self):
        if self.options["cell geometry"] == "cylindrical":
            return pybamm.battery_geometry(
                options=self.options, form_factor="cylindrical"
            )
        else:
            return pybamm.battery_geometry(options=self.options)

    @property
    def default_var_pts(self):
        base_var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_p": 20,
            "r_n_prim": 20,
            "r_p_prim": 20,
            "r_n_sec": 20,
            "r_p_sec": 20,
            "y": 10,
            "z": 10,
            "R_n": 30,
            "R_p": 30,
            "R_n_prim": 30,
            "R_p_prim": 30,
            "R_n_sec": 30,
            "R_p_sec": 30,
        }
        # Reduce the default points for 2D current collectors
        if self.options["dimensionality"] == 2:
            base_var_pts.update({"x_n": 10, "x_s": 10, "x_p": 10})
        return base_var_pts

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "negative primary particle": pybamm.Uniform1DSubMesh,
            "positive primary particle": pybamm.Uniform1DSubMesh,
            "negative secondary particle": pybamm.Uniform1DSubMesh,
            "positive secondary particle": pybamm.Uniform1DSubMesh,
            "negative particle size": pybamm.Uniform1DSubMesh,
            "positive particle size": pybamm.Uniform1DSubMesh,
            "negative primary particle size": pybamm.Uniform1DSubMesh,
            "positive primary particle size": pybamm.Uniform1DSubMesh,
            "negative secondary particle size": pybamm.Uniform1DSubMesh,
            "positive secondary particle size": pybamm.Uniform1DSubMesh,
        }
        if self.options["dimensionality"] == 0:
            base_submeshes["current collector"] = pybamm.SubMesh0D
        elif self.options["dimensionality"] == 1:
            base_submeshes["current collector"] = pybamm.Uniform1DSubMesh

        elif self.options["dimensionality"] == 2:
            base_submeshes["current collector"] = pybamm.ScikitUniform2DSubMesh
        elif self.options["dimensionality"] == 3:
            base_submeshes["current collector"] = pybamm.SubMesh0D
            geom_type = self.options.get("cell geometry", "pouch")
            if geom_type == "pouch":
                base_submeshes["cell"] = pybamm.ScikitFemGenerator3D(
                    geom_type="pouch", h="0.1"
                )
            elif geom_type == "cylindrical":
                base_submeshes["cell"] = pybamm.ScikitFemGenerator3D(
                    geom_type="cylinder", h="0.1"
                )
        return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume(),
            "negative particle": pybamm.FiniteVolume(),
            "positive particle": pybamm.FiniteVolume(),
            "negative primary particle": pybamm.FiniteVolume(),
            "positive primary particle": pybamm.FiniteVolume(),
            "negative secondary particle": pybamm.FiniteVolume(),
            "positive secondary particle": pybamm.FiniteVolume(),
            "negative particle size": pybamm.FiniteVolume(),
            "positive particle size": pybamm.FiniteVolume(),
            "negative primary particle size": pybamm.FiniteVolume(),
            "positive primary particle size": pybamm.FiniteVolume(),
            "negative secondary particle size": pybamm.FiniteVolume(),
            "positive secondary particle size": pybamm.FiniteVolume(),
        }
        if self.options["dimensionality"] == 0:
            # 0D submesh - use base spatial method
            base_spatial_methods["current collector"] = (
                pybamm.ZeroDimensionalSpatialMethod()
            )
        elif self.options["dimensionality"] == 1:
            base_spatial_methods["current collector"] = pybamm.FiniteVolume()
        elif self.options["dimensionality"] == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement()
        elif self.options["dimensionality"] == 3:
            base_spatial_methods["current collector"] = (
                pybamm.ZeroDimensionalSpatialMethod()
            )
            base_spatial_methods["cell"] = pybamm.ScikitFiniteElement3D()
        return base_spatial_methods

    #: Option values the model needs; a supplied value must match.
    fixed_options = {}
    #: Option values the model prefers; the caller may override them.
    default_options = {}

    def _dependent_default_options(self, supplied):
        """Return model defaults that depend on the options the caller gave.

        Relying on these is deprecated, so each one that is used emits a
        :class:`pybamm.OptionDefaultDeprecationWarning`.

        Parameters
        ----------
        supplied : dict
            The options given by the caller.

        Returns
        -------
        dict
            Defaults for options the caller did not give.
        """
        return {}

    def _check_fixed_options(self, options):
        """Check that options agree with the model's fixed options.

        Parameters
        ----------
        options : dict
            The options given by the caller, or processed options.

        Raises
        ------
        pybamm.OptionError
            If an option does not match a fixed option of the model.
        """
        working_electrode = options.get(
            "working electrode", self.fixed_options.get("working electrode", "both")
        )
        for option, value in self.fixed_options.items():
            if option in options and not option_values_match(
                option, options[option], value, working_electrode
            ):
                raise pybamm.OptionError(
                    f"{type(self).__name__} requires '{option}' to be {value!r}, "
                    f"not {options[option]!r}."
                )

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        # if extra_options is a dict then process it into a BatteryModelOptions
        # this does not catch cases that subclass the dict type
        # so other submodels can pass in their own options class if needed
        if extra_options is None or type(extra_options) in (dict, LegacyOptions):
            supplied = dict(extra_options or {})
            self._check_fixed_options(supplied)
            model_legacy_defaults = self._dependent_default_options(supplied)
            options = BatteryModelOptions(
                {
                    **self.default_options,
                    **model_legacy_defaults,
                    **self.fixed_options,
                    **supplied,
                },
                legacy=isinstance(extra_options, LegacyOptions),
                warn_legacy_defaults=False,
            )
            legacy_defaults = options._legacy_defaults
        else:
            options = extra_options
            legacy_defaults = {}
            model_legacy_defaults = {}
            self._check_fixed_options(options)

        # Options that are incompatible with models
        if (
            isinstance(self, pybamm.lithium_ion.BaseModel)
            and options["convection"] != "none"
        ):
            raise pybamm.OptionError(
                "convection not implemented for lithium-ion models"
            )
        if isinstance(self, pybamm.lithium_ion.SPMe) and options[
            "electrolyte conductivity"
        ] not in [
            "default",
            "composite",
            "integrated",
        ]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPMe".format(
                    options["electrolyte conductivity"]
                )
            )
        if isinstance(self, pybamm.lithium_ion.SPM) and (
            "distribution" in options["particle size"]
            and options["surface form"] == "false"
        ):
            raise pybamm.OptionError(
                "surface form must be 'algebraic' or 'differential' if "
                " 'particle size' contains a 'distribution'"
            )
        if isinstance(self, pybamm.lead_acid.BaseModel):
            if options["thermal"] != "isothermal" and options["dimensionality"] != 0:
                raise pybamm.OptionError(
                    "Lead-acid models can only have thermal "
                    "effects if dimensionality is 0."
                )

            def leaves(option):
                return {leaf for _, leaf in iter_option_leaves(option, options[option])}

            if leaves("SEI") != {"none"} or options["SEI film resistance"] != "none":
                raise pybamm.OptionError("Lead-acid models cannot have SEI formation")
            if leaves("lithium plating") != {"none"}:
                raise pybamm.OptionError("Lead-acid models cannot have lithium plating")
            if "MSMR" in leaves("open-circuit potential"):
                raise pybamm.OptionError(
                    "Lead-acid models cannot use the MSMR open-circuit potential model"
                )

        if (
            isinstance(self, pybamm.lead_acid.LOQS)
            and options["surface form"] == "false"
            and options["hydrolysis"] == "true"
        ):
            raise pybamm.OptionError(
                f"must use surface formulation to solve {self!s} with hydrolysis"
            )
        _warn_legacy_defaults(legacy_defaults)
        _warn_legacy_defaults(model_legacy_defaults)
        self._options = options
        # rebuild whenever options are (re)assigned.
        # No-op unless the subclass overrides ``_rebuild_param``.
        self._rebuild_param()

    def set_standard_output_variables(self):
        # Time
        self.variables.update(
            {
                "Time [s]": pybamm.t,
                "Time [min]": pybamm.t / 60,
                "Time [h]": pybamm.t / 3600,
            }
        )

        # Spatial
        var = pybamm.standard_spatial_vars
        self.variables.update(
            {"x [m]": var.x, "x_n [m]": var.x_n, "x_s [m]": var.x_s, "x_p [m]": var.x_p}
        )
        if self.options["dimensionality"] == 1:
            self.variables.update({"z [m]": var.z})
        elif self.options["dimensionality"] == 2:
            self.variables.update({"y [m]": var.y, "z [m]": var.z})

    def build_model_equations(self):
        # Set model equations
        for submodel_name, submodel in self.submodels.items():
            pybamm.logger.verbose(
                f"Setting rhs for {submodel_name} submodel ({self.name})"
            )

            submodel.set_rhs(self.variables)
            pybamm.logger.verbose(
                f"Setting algebraic for {submodel_name} submodel ({self.name})"
            )

            submodel.set_algebraic(self.variables)
            pybamm.logger.verbose(
                f"Setting boundary conditions for {submodel_name} submodel ({self.name})"
            )

            submodel.set_boundary_conditions(self.variables)
            pybamm.logger.verbose(
                f"Setting initial conditions for {submodel_name} submodel ({self.name})"
            )
            submodel.set_initial_conditions(self.variables)
            submodel.add_events_from(self.variables)
            pybamm.logger.verbose(f"Updating {submodel_name} submodel ({self.name})")
            self.update(submodel)
            self.check_no_repeated_keys()

    def _build_model(self):
        if self._built:
            raise pybamm.ModelError(
                """Model already built. If you are adding a new submodel, try using
                `model.update` instead."""
            )

        pybamm.logger.info(f"Start building {self.name}")

        if self._built_fundamental is False:
            self.build_fundamental()

        # Register the voltage state Variable before coupled variables,
        # so submodels that need V in get_coupled_variables can find it.
        if self.options["voltage as a state"] == "true":
            self._register_voltage_variable()

        self.build_coupled_variables()

        # Now that the expression is available, add the algebraic constraint
        # before equations are built.
        if self.options["voltage as a state"] == "true":
            self._constrain_voltage_to_expression()

        self.build_model_equations()

    def build_model(self):
        # Build model variables and equations
        self._build_model()

        # Set battery specific variables
        pybamm.logger.debug(f"Setting voltage variables ({self.name})")
        self.set_voltage_variables()

        pybamm.logger.debug(f"Setting SoC variables ({self.name})")
        self.set_soc_variables()

        pybamm.logger.debug(f"Setting degradation variables ({self.name})")
        self.set_degradation_variables()
        self.set_summary_variables()

        self._built = True
        pybamm.logger.info(f"Finish building {self.name}")

    @property
    def summary_variables(self):
        return self._summary_variables

    @summary_variables.setter
    def summary_variables(self, value):
        """
        Set summary variables

        Parameters
        ----------
        value : list of strings
            Names of the summary variables. Must all be in self.variables.
        """
        for var in value:
            if var not in self.variables:
                raise KeyError(
                    f"No cycling variable defined for summary variable '{var}'"
                )
        self._summary_variables = value

    def set_summary_variables(self):
        self._summary_variables = []

    def get_intercalation_kinetics(self, domain):
        options = getattr(self.options, domain)
        if options["intercalation kinetics"] == "symmetric Butler-Volmer":
            return pybamm.kinetics.SymmetricButlerVolmer
        elif options["intercalation kinetics"] == "asymmetric Butler-Volmer":
            return pybamm.kinetics.AsymmetricButlerVolmer
        elif options["intercalation kinetics"] == "linear":
            return pybamm.kinetics.Linear
        elif options["intercalation kinetics"] == "Marcus":
            return pybamm.kinetics.Marcus
        elif options["intercalation kinetics"] == "Marcus-Hush-Chidsey":
            return pybamm.kinetics.MarcusHushChidsey
        elif options["intercalation kinetics"] == "MSMR":
            return pybamm.kinetics.MSMRButlerVolmer

    def get_inverse_intercalation_kinetics(self, domain):
        options = getattr(self.options, domain)
        if options["intercalation kinetics"] == "symmetric Butler-Volmer":
            return pybamm.kinetics.InverseButlerVolmer
        elif options["intercalation kinetics"] == "linear":
            return pybamm.kinetics.InverseLinear
        else:
            raise pybamm.OptionError(
                "Inverse kinetics are only implemented for symmetric Butler-Volmer. "
                "Use option {'surface form': 'algebraic'} to use forward kinetics "
                "instead."
            )

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            model = pybamm.external_circuit.ExplicitCurrentControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "voltage":
            model = pybamm.external_circuit.VoltageFunctionControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "power":
            model = pybamm.external_circuit.PowerFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential power":
            model = pybamm.external_circuit.PowerFunctionControl(
                self.param, self.options, "differential"
            )
        elif self.options["operating mode"] == "explicit power":
            model = pybamm.external_circuit.ExplicitPowerControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "differential"
            )
        elif self.options["operating mode"] == "explicit resistance":
            model = pybamm.external_circuit.ExplicitResistanceControl(
                self.param, self.options
            )
        elif self.options["operating mode"] == "CCCV":
            model = pybamm.external_circuit.CCCVFunctionControl(
                self.param, self.options
            )
        elif callable(self.options["operating mode"]):
            model = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"], self.options
            )
        self.submodels["external circuit"] = model
        self.submodels["discharge and throughput variables"] = (
            pybamm.external_circuit.DischargeThroughput(self.param, self.options)
        )

    def set_transport_efficiency_submodels(self):
        if self.options["transport efficiency"] == "Bruggeman":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.Bruggeman(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.Bruggeman(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "tortuosity factor":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.TortuosityFactor(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.TortuosityFactor(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "ordered packing":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.OrderedPacking(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.OrderedPacking(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "hyperbola of revolution":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.HyperbolaOfRevolution(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.HyperbolaOfRevolution(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "overlapping spheres":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.OverlappingSpheres(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.OverlappingSpheres(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "random overlapping cylinders":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.RandomOverlappingCylinders(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.RandomOverlappingCylinders(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "heterogeneous catalyst":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.HeterogeneousCatalyst(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.HeterogeneousCatalyst(
                    self.param, "Electrode", self.options
                )
            )
        elif self.options["transport efficiency"] == "cation-exchange membrane":
            self.submodels["electrolyte transport efficiency"] = (
                pybamm.transport_efficiency.CationExchangeMembrane(
                    self.param, "Electrolyte", self.options
                )
            )
            self.submodels["electrode transport efficiency"] = (
                pybamm.transport_efficiency.CationExchangeMembrane(
                    self.param, "Electrode", self.options
                )
            )

    def set_thermal_submodel(self):
        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal
        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped
        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                thermal_submodel = pybamm.thermal.Lumped
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector1D
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector2D
        elif (
            self.options["thermal"] == "x-full" and self.options["dimensionality"] == 0
        ):
            thermal_submodel = pybamm.thermal.pouch_cell.OneDimensionalX

        x_average = getattr(self, "x_average", False)
        self.submodels["thermal"] = thermal_submodel(
            self.param, self.options, x_average
        )

    def set_surface_temperature_submodel(self):
        if self.options["surface temperature"] == "ambient":
            submodel = pybamm.thermal.surface.Ambient
        elif self.options["surface temperature"] == "lumped":
            submodel = pybamm.thermal.surface.Lumped
        self.submodels["surface temperature"] = submodel(self.param, self.options)

    def set_current_collector_submodel(self):
        if self.options["current collector"] in ["uniform"]:
            submodel = pybamm.current_collector.Uniform(self.param)
        elif self.options["current collector"] == "potential pair":
            if self.options["dimensionality"] == 1:
                submodel = pybamm.current_collector.PotentialPair1plus1D(self.param)
            elif self.options["dimensionality"] == 2:
                submodel = pybamm.current_collector.PotentialPair2plus1D(self.param)
        self.submodels["current collector"] = submodel

    def set_interface_utilisation_submodel(self):
        for domain in ["negative", "positive"]:
            Domain = domain.capitalize()
            util = getattr(self.options, domain)["interface utilisation"]
            if util == "full":
                submodel = pybamm.interface_utilisation.Full(
                    self.param, domain, self.options
                )
            elif util == "constant":
                submodel = pybamm.interface_utilisation.Constant(
                    self.param, domain, self.options
                )
            elif util == "current-driven":
                if self.options.electrode_types[domain] == "planar":
                    reaction_loc = "interface"
                elif self.x_average:
                    reaction_loc = "x-average"
                else:
                    reaction_loc = "full electrode"
                submodel = pybamm.interface_utilisation.CurrentDriven(
                    self.param, domain, self.options, reaction_loc
                )
            self.submodels[f"{Domain} interface utilisation"] = submodel

    def _register_voltage_variable(self):
        V = pybamm.Variable("Voltage [V]")
        self.variables["Voltage [V]"] = V
        self.initial_conditions[V] = self.param.ocv_init

    def _constrain_voltage_to_expression(self):
        V = self.variables["Voltage [V]"]
        V_expr = self.variables.get("Voltage expression [V]")
        if V_expr is None:
            raise pybamm.ModelError(
                "'voltage as a state' requires the model to define "
                "'Voltage expression [V]'"
            )
        self.algebraic[V] = V - V_expr

    def set_voltage_variables(self):
        if self.options.negative["particle phases"] == "1":
            # Only one phase, no need to distinguish between
            # "primary" and "secondary"
            phase_n = ""
        else:
            # add a space so that we can use "" or (e.g.) "primary " interchangeably
            # when naming variables
            phase_n = "primary "
        if self.options.positive["particle phases"] == "1":
            phase_p = ""
        else:
            phase_p = "primary "

        ocp_surf_n_av = self.variables[
            f"X-averaged negative electrode {phase_n}open-circuit potential [V]"
        ]
        ocp_surf_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}open-circuit potential [V]"
        ]
        ocp_n_bulk = self.variables[
            f"Negative electrode {phase_n}bulk open-circuit potential [V]"
        ]
        ocp_p_bulk = self.variables[
            f"Positive electrode {phase_p}bulk open-circuit potential [V]"
        ]
        eta_particle_n = self.variables[
            f"Negative {phase_n}particle concentration overpotential [V]"
        ]
        eta_particle_p = self.variables[
            f"Positive {phase_p}particle concentration overpotential [V]"
        ]

        ocv_surf = ocp_surf_p_av - ocp_surf_n_av
        ocv_bulk = ocp_p_bulk - ocp_n_bulk

        eta_particle = eta_particle_p - eta_particle_n

        # overpotentials
        if self.options.electrode_types["negative"] == "planar":
            eta_r_n_av = self.variables[
                "Lithium metal interface reaction overpotential [V]"
            ]
        else:
            eta_r_n_av = self.variables[
                f"X-averaged negative electrode {phase_n}reaction overpotential [V]"
            ]
        eta_r_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}reaction overpotential [V]"
        ]
        eta_r_av = eta_r_p_av - eta_r_n_av

        delta_phi_s_n_av = self.variables[
            "X-averaged negative electrode ohmic losses [V]"
        ]
        delta_phi_s_p_av = self.variables[
            "X-averaged positive electrode ohmic losses [V]"
        ]
        delta_phi_s_av = delta_phi_s_p_av - delta_phi_s_n_av

        # SEI film overpotential
        if self.options.electrode_types["negative"] == "planar":
            eta_sei_n_av = self.variables[
                "Negative electrode SEI film overpotential [V]"
            ]
        else:
            eta_sei_n_av = self.variables[
                f"X-averaged negative electrode {phase_n}SEI film overpotential [V]"
            ]
        eta_sei_p_av = self.variables[
            f"X-averaged positive electrode {phase_p}SEI film overpotential [V]"
        ]
        eta_sei_av = eta_sei_n_av + eta_sei_p_av

        # TODO: add current collector losses to the voltage in 3D

        self.variables.update(
            {
                "Surface open-circuit voltage [V]": ocv_surf,
                "Bulk open-circuit voltage [V]": ocv_bulk,
                "Particle concentration overpotential [V]": eta_particle,
                "X-averaged reaction overpotential [V]": eta_r_av,
                "X-averaged SEI film overpotential [V]": eta_sei_av,
                "X-averaged solid phase ohmic losses [V]": delta_phi_s_av,
            }
        )

        # Battery-wide variables
        V = self.variables["Voltage [V]"]
        eta_e_av = self.variables["X-averaged electrolyte ohmic losses [V]"]
        eta_c_av = self.variables["X-averaged concentration overpotential [V]"]
        num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )
        self.variables.update(
            {
                "Battery open-circuit voltage [V]": ocv_bulk * num_cells,
                "Battery negative electrode bulk open-circuit potential [V]": ocp_n_bulk
                * num_cells,
                "Battery positive electrode bulk open-circuit potential [V]": ocp_p_bulk
                * num_cells,
                "Battery particle concentration overpotential [V]": eta_particle
                * num_cells,
                "Battery negative particle concentration overpotential [V]"
                "": eta_particle_n * num_cells,
                "Battery positive particle concentration overpotential [V]"
                "": eta_particle_p * num_cells,
                "X-averaged battery reaction overpotential [V]": eta_r_av * num_cells,
                "X-averaged battery negative reaction overpotential [V]": eta_r_n_av
                * num_cells,
                "X-averaged battery positive reaction overpotential [V]": eta_r_p_av
                * num_cells,
                "X-averaged battery solid phase ohmic losses [V]": delta_phi_s_av
                * num_cells,
                "X-averaged battery negative solid phase ohmic losses [V]"
                "": delta_phi_s_n_av * num_cells,
                "X-averaged battery positive solid phase ohmic losses [V]"
                "": delta_phi_s_p_av * num_cells,
                "X-averaged battery electrolyte ohmic losses [V]": eta_e_av * num_cells,
                "X-averaged battery concentration overpotential [V]": eta_c_av
                * num_cells,
                "Battery voltage [V]": V * num_cells,
            }
        )

        # Calculate equivalent resistance of an OCV-R Equivalent Circuit Model
        # ECM overvoltage is OCV minus voltage
        v_ecm = ocv_bulk - V

        # Hack to avoid division by zero if i_cc is exactly zero
        # If i_cc is zero, i_cc_not_zero becomes 1. But multiplying by sign(i_cc) makes
        # the local resistance 'zero' (really, it's not defined when i_cc is zero)
        def x_not_zero(x):
            return ((x > 0) + (x < 0)) * x + (x >= 0) * (x <= 0)

        i_cc = self.variables["Current collector current density [A.m-2]"]
        i_cc_not_zero = x_not_zero(i_cc)
        A_cc = self.param.A_cc

        self.variables.update(
            {
                "Local ECM resistance [Ohm]": pybamm.sign(i_cc)
                * v_ecm
                / (i_cc_not_zero * A_cc),
            }
        )

        # Cut-off voltage
        self.events.append(
            pybamm.Event(
                "Minimum voltage [V]",
                V - self.param.voltage_low_cut,
                pybamm.EventType.TERMINATION,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage [V]",
                self.param.voltage_high_cut - V,
                pybamm.EventType.TERMINATION,
            )
        )

        # Cut-off open-circuit voltage (for event switch with casadi 'fast with events'
        # mode)
        tol = 0.1
        self.events.append(
            pybamm.Event(
                "Minimum voltage switch [V]",
                V - (self.param.voltage_low_cut - tol),
                pybamm.EventType.SWITCH,
            )
        )
        self.events.append(
            pybamm.Event(
                "Maximum voltage switch [V]",
                V - (self.param.voltage_high_cut + tol),
                pybamm.EventType.SWITCH,
            )
        )

        # Power and resistance
        I = self.variables["Current [A]"]
        I_not_zero = x_not_zero(I)
        self.variables.update(
            {
                "Terminal power [W]": I * V,
                "Power [W]": I * V,
                "Resistance [Ohm]": pybamm.sign(I) * V / I_not_zero,
            }
        )

    def set_degradation_variables(self):
        """
        Set variables that quantify degradation.
        This function is overriden by the base battery models
        """

    def set_soc_variables(self):
        """
        Set variables relating to the state of charge.
        This function is overriden by the base battery models
        """

    def save_model(self, filename=None, mesh=None, variables=None):
        """
        Write out a discretised model to a JSON file

        Parameters
        ----------
        filename: str, optional
        The desired name of the JSON file. If no name is provided, one will be created
        based on the model name, and the current datetime.
        """
        if variables and not mesh:
            raise ValueError(
                "Serialisation: Please provide the mesh if variables are required"
            )

        Serialise().save_model(self, filename=filename, mesh=mesh, variables=variables)
