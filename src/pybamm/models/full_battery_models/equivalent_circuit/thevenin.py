import pybamm

from .ecm_model_options import NaturalNumberOption, OperatingModes


class Thevenin(pybamm.BaseModel):
    """
    The classical Thevenin Equivalent Circuit Model of a battery as
    described in, for example, :footcite:t:`Barletta2022thevenin`.

    This equivalent circuit model consists of an OCV element, a resistor
    element, and a number of RC elements (by default 1). The model is
    coupled to two lumped thermal models, one for the cell and
    one for the surrounding jig. Heat generation terms for each element
    follow equation (1) of :footcite:t:`Nieto2012`.

    Parameters
    ----------
    name : str, optional
        The name of the model. The default is
        "Thevenin Equivalent Circuit Model".
    options : dict, optional
        A dictionary of options to be passed to the model. The default is None.
        Possible options are:

            * "number of rc elements" : str
                The number of RC elements to be added to the model. The default is 1.
            * "calculate discharge energy": str
                Whether to calculate the discharge energy, throughput energy and
                throughput capacity in addition to discharge capacity. Must be one of
                "true" or "false". "false" is the default, since calculating discharge
                energy can be computationally expensive for simple models like SPM.
            * "diffusion element" : str
                Whether to include the diffusion element to the model. Must be one of
                "true" or "false". "false" is the default.
            * "operating mode" : str
                Sets the operating mode for the model. This determines how the current
                is set. Can be:

                - "current" (default) : the current is explicity supplied
                - "voltage"/"power"/"resistance" : solve an algebraic equation for \
                    current such that voltage/power/resistance is correct
                - "differential power"/"differential resistance" : solve a \
                    differential equation for the power or resistance
                - "CCCV": a special implementation of the common constant-current \
                    constant-voltage charging protocol, via an ODE for the current
                - callable : if a callable is given as this option, the function \
                    defines the residual of an algebraic equation. The applied current \
                    will be solved for such that the algebraic constraint is satisfied.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).

    Examples
    --------
    >>> model = pybamm.equivalent_circuit.Thevenin()
    >>> model.name
    'Thevenin Equivalent Circuit Model'

    """

    def __init__(
        self, name="Thevenin Equivalent Circuit Model", options=None, build=True
    ):
        super().__init__(name)

        self.set_options(options)
        self.param = pybamm.EcmParameters()
        self.element_counter = 0

        self.set_standard_output_variables()
        self.set_submodels(build)

    def set_options(self, extra_options=None):
        possible_options = {
            "calculate discharge energy": ["false", "true"],
            "diffusion element": ["false", "true"],
            "operating mode": OperatingModes("current"),
            "number of rc elements": NaturalNumberOption(1),
        }

        default_options = {
            name: options[0] for name, options in possible_options.items()
        }

        extra_options = extra_options or {}

        options = pybamm.FuzzyDict(default_options)
        for name, opt in extra_options.items():
            if name in default_options:
                options[name] = opt
            else:
                raise pybamm.OptionError(
                    f"Option '{name}' not recognised. Best matches are {options.get_best_matches(name)}"
                )

        for opt, value in options.items():
            if value not in possible_options[opt]:
                raise pybamm.OptionError(
                    f"Option '{opt}' must be one of {possible_options[opt]}. Got '{value}' instead."
                )

        self.options = options

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
        elif self.options["operating mode"] == "resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "algebraic"
            )
        elif self.options["operating mode"] == "differential resistance":
            model = pybamm.external_circuit.ResistanceFunctionControl(
                self.param, self.options, "differential"
            )
        elif self.options["operating mode"] == "CCCV":
            model = pybamm.external_circuit.CCCVFunctionControl(
                self.param, self.options
            )
        elif callable(self.options["operating mode"]):
            model = pybamm.external_circuit.FunctionControl(
                self.param,
                self.options["operating mode"],
                self.options,
                control="differential",
            )
        self.submodels["external circuit"] = model

    def set_ocv_submodel(self):
        self.submodels["Open-circuit voltage"] = (
            pybamm.equivalent_circuit_elements.OCVElement(self.param, self.options)
        )

    def set_resistor_submodel(self):
        name = "Element-0 (Resistor)"
        self.submodels[name] = pybamm.equivalent_circuit_elements.ResistorElement(
            self.param, self.options
        )
        self.element_counter += 1

    def set_rc_submodels(self):
        number_of_rc_elements = self.options["number of rc elements"]

        for _ in range(number_of_rc_elements):
            name = f"Element-{self.element_counter} (RC)"
            self.submodels[name] = pybamm.equivalent_circuit_elements.RCElement(
                self.param, self.element_counter, self.options
            )
            self.element_counter += 1

    def set_diffusion_submodel(self):
        if self.options["diffusion element"] == "false":
            self.submodels["Diffusion"] = (
                pybamm.equivalent_circuit_elements.NoDiffusion(self.param, self.options)
            )
        elif self.options["diffusion element"] == "true":
            self.submodels["Diffusion"] = (
                pybamm.equivalent_circuit_elements.DiffusionElement(
                    self.param, self.options
                )
            )

    def set_thermal_submodel(self):
        self.submodels["Thermal"] = pybamm.equivalent_circuit_elements.ThermalSubModel(
            self.param, self.options
        )

    def set_voltage_submodel(self):
        self.submodels["Voltage"] = pybamm.equivalent_circuit_elements.VoltageModel(
            self.param, self.options
        )

    def set_submodels(self, build):
        self.set_external_circuit_submodel()
        self.set_ocv_submodel()
        self.set_resistor_submodel()
        self.set_rc_submodels()
        self.set_diffusion_submodel()
        self.set_thermal_submodel()
        self.set_voltage_submodel()

        self.summary_variables = []

        if build:
            self.build_model()

    def set_standard_output_variables(self):
        # Time
        self.variables.update(
            {
                "Time [s]": pybamm.t,
                "Time [min]": pybamm.t / 60,
                "Time [h]": pybamm.t / 3600,
            }
        )

    def build_model(self):
        # Build model variables and equations
        self._build_model()

        self._built = True
        pybamm.logger.info(f"Finished building {self.name}")

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues("ECM_Example")

    @property
    def default_quick_plot_variables(self):
        return [
            "Current [A]",
            ["Voltage [V]", "Open-circuit voltage [V]"],
            "SoC",
            "Power [W]",
            [
                "Cell temperature [degC]",
                "Jig temperature [degC]",
                "Ambient temperature [degC]",
            ],
            [
                "Total heat generation [W]",
                "Reversible heat generation [W]",
                "Irreversible heat generation [W]",
            ],
        ]

    @property
    def default_var_pts(self):
        x = pybamm.SpatialVariable(
            "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
        )
        return {x: 20}

    @property
    def default_geometry(self):
        x = pybamm.SpatialVariable(
            "x ECMD", domain=["ECMD particle"], coord_sys="Cartesian"
        )
        return {"ECMD particle": {x: {"min": 0, "max": 1}}}

    @property
    def default_submesh_types(self):
        return {"ECMD particle": pybamm.Uniform1DSubMesh}

    @property
    def default_spatial_methods(self):
        return {"ECMD particle": pybamm.FiniteVolume()}
