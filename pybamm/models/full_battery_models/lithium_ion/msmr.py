import pybamm
from .dfn import DFN


class MSMR(DFN):
    def __init__(self, options=None, name="MSMR", build=True):
        # Necessary/default options
        options = options or {}
        if "number of MSMR reactions" not in options:
            raise pybamm.OptionError(
                "number of MSMR reactions must be specified for MSMR"
            )
        if (
            "open-circuit potential" in options
            and options["open-circuit potential"] != "MSMR"
        ):
            raise pybamm.OptionError(
                "'open-circuit potential' must be 'MSMR' for MSMR not '{}'".format(
                    options["open-circuit potential"]
                )
            )
        elif "particle" in options and options["particle"] != "MSMR":
            raise pybamm.OptionError(
                "'particle' must be 'MSMR' for MSMR not '{}'".format(
                    options["particle"]
                )
            )
        elif (
            "intercalation kinetics" in options
            and options["intercalation kinetics"] != "MSMR"
        ):
            raise pybamm.OptionError(
                "'intercalation kinetics' must be 'MSMR' for MSMR not '{}'".format(
                    options["intercalation kinetics"]
                )
            )
        else:
            options.update(
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "intercalation kinetics": "MSMR",
                }
            )
        super().__init__(options=options, name=name)

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues("MSMR_Example")
