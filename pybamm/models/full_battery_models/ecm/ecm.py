import pybamm

from pybamm import EcmParameters
from pybamm import OcvElement
from pybamm import ResistorElement
from pybamm import RcElement
from pybamm import ThermalSubModel
from pybamm import AnodeVoltageEstimator
from pybamm import VoltageModel


class EquivalentCircuitModel(pybamm.BaseModel):
    def __init__(self, name="Equivalent Circuit Model", options=None, build=True):
        super().__init__(name)

        self.set_options(options)
        self.param = EcmParameters()
        self.element_counter = 0

        self.set_submodels(build)

    def set_options(self, extra_options=None):

        possible_options = {
            "calculate discharge energy": ["false", "true"],
            "operating mode": [
                "current",
                "voltage",
                "power",
                "differential power",
                "explicit power",
                "resistance",
                "differential resistance",
                "explicit resistance",
                "CCCV",
            ],
            "include resistor": ["true", "false"],
            "number of rc elements": [2, 1, 3, 4],
            "external submodels": [[]],
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
                    "Option '{}' not recognised. Best matches are {}".format(
                        name, options.get_best_matches(name)
                    )
                )

        self.ecm_options = options

        # Hack to deal with submodels requiring electrochemical model
        # options
        self.options = pybamm.BatteryModelOptions({})
        self.options["calculate discharge energy"] = self.ecm_options[
            "calculate discharge energy"
        ]
        self.options["operating mode"] = self.ecm_options["operating mode"]

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
                self.param, self.options, "differential without max"
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
                self.param, self.options, "differential without max"
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
                self.param,
                self.options["operating mode"],
                self.options,
                control="differential without max",
            )
        self.submodels["external circuit"] = model

    def set_ocv_submodel(self):
        self.submodels["Open circuit voltage"] = OcvElement(
            self.param, self.ecm_options
        )

    def set_resistor_submodel(self):

        include_resistor = self.ecm_options["include resistor"]

        if include_resistor == "true":

            name = f"Element-{self.element_counter} (Resistor)"
            self.submodels[name] = ResistorElement(
                self.param, self.element_counter, self.ecm_options
            )
            self.element_counter += 1

    def set_rc_submodels(self):
        number_of_rc_elements = self.ecm_options["number of rc elements"]

        for _ in range(number_of_rc_elements):
            name = f"Element-{self.element_counter} (RC)"
            self.submodels[name] = RcElement(
                self.param, self.element_counter, self.ecm_options
            )
            self.element_counter += 1

    def set_thermal_submodel(self):
        self.submodels["Thermal"] = ThermalSubModel(self.param, self.ecm_options)

    def set_anode_voltage_submodel(self):
        self.submodels["Anode voltage"] = AnodeVoltageEstimator(
            self.param, self.ecm_options
        )

    def set_voltage_submodel(self):
        self.submodels["Voltage"] = VoltageModel(self.param, self.ecm_options)

    def set_submodels(self, build):
        self.set_external_circuit_submodel()
        self.set_ocv_submodel()
        self.set_resistor_submodel()
        self.set_rc_submodels()
        self.set_thermal_submodel()
        self.set_anode_voltage_submodel()
        self.set_voltage_submodel()

        self.summary_variables = []

        if build:
            self.build_model()

    def build_model(self):

        # Build model variables and equations
        self._build_model()

        self._built = True
        pybamm.logger.info("Finished building {}".format(self.name))
