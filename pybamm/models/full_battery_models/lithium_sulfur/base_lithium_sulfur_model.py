#
# Lithium-sulfur base model class
#
import pybamm


class BaseModel(pybamm.BaseModel):
    """
    Base model for lithium-sulfur battery models.

    Note: this should really extend :class:`pybamm.BaseBatteryModel`, but the
    majority of options aren't compatible with the lithium-sulfur models, and
    the lithium-sulfur models don't use the submodel structure. For now it is
    simplest to just extend :class:`pybamm.BaseModel`.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, options=None, name="Unnamed lithium-sulfur model"):
        super().__init__(name)
        self.options = options
        self.param = pybamm.LithiumSulfurParameters()

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """

        # Set variable for the terminal voltage
        V = pybamm.Variable("Terminal voltage [V]")
        self.variables.update({"Terminal voltage [V]": V})

        # Set up current operated. Here the current is just provided as a
        # parameter
        if self.options["operating mode"] == "current":
            I = pybamm.Parameter("Current function [A]")
            self.variables.update({"Current [A]": I})

        # If the the operating mode of the simulation is "with experiment" the
        # model option "operating mode" is the callable function
        # 'constant_current_constant_voltage_constant_power'
        elif callable(self.options["operating mode"]):
            # For the experiment we solve an extra algebraic equation
            # to determine the current
            I = pybamm.Variable("Current [A]")
            self.variables.update({"Current [A]": I})
            control_function = self.options["operating mode"]

            # 'constant_current_constant_voltage_constant_power' is a function
            # of current and voltage via the variables dict (see pybamm.Simulation)
            self.algebraic = {I: control_function(self.variables)}
            self.initial_conditions[I] = pybamm.Parameter("Current function [A]")

        # Add variable for discharge capacity
        Q = pybamm.Variable("Discharge capacity [A.h]")
        self.variables.update({"Discharge capacity [A.h]": Q})
        self.rhs.update({Q: I * self.param.timescale / 3600})
        self.initial_conditions.update({Q: pybamm.Scalar(0)})

    @property
    def default_parameter_values(self):
        # TODO: separate parameters out by component and create a parameter set
        # that can be called (see pybamm/parameters/parameter_sets.py)
        return pybamm.ParameterValues(values="lithium-sulfur/parameters.csv")

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, extra_options):
        default_options = {"operating mode": "current"}
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

        # check operating mode is allowed
        if not (
            options["operating mode"] in ["current"]
            or callable(options["operating mode"])
        ):
            raise pybamm.OptionError(
                "operating mode '{}' not recognised".format(options["operating mode"])
            )

        self._options = options

    def new_copy(self, options=None):
        "Create an empty copy with identical options, or new options if specified"
        options = options or self.options
        new_model = self.__class__(options=options, name=self.name)
        new_model.use_jacobian = self.use_jacobian
        new_model.convert_to_format = self.convert_to_format
        new_model.timescale = self.timescale
        new_model.length_scales = self.length_scales
        return new_model
