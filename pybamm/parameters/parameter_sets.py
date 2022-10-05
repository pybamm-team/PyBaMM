#
# Parameter sets from papers
# This file is being deprecated and replaced with defined parameter sets directly
# in python files in the input/parameters folder. The Ai2020 parameter set is
# kept here for now as it is used in the tests, but will be removed in the future.
#
import warnings


class ParameterSets:
    def __getattribute__(self, name):
        # kept for testing for now
        if name == "Ai2020":
            out = {
                "chemistry": "lithium_ion",
                "cell": "Enertech_Ai2020",
                "negative electrode": "graphite_Ai2020",
                "separator": "separator_Ai2020",
                "positive electrode": "lico2_Ai2020",
                "electrolyte": "lipf6_Enertech_Ai2020",
                "experiment": "1C_discharge_from_full_Ai2020",
                "sei": "example",
                "citation": "Ai2019",
            }
        # For backwards compatibility, parameter sets that used to be defined in this
        # file now return the name as a string, which will load the same parameter
        # set as before when passed to `ParameterValues`
        elif name in [
            "Chen2020",
            "Chen2020_composite",
            "Ecker2015",
            "Marquis2019",
            "Mohtat2020",
            "NCA_Kim2011",
            "OKane2022",
            "ORegan2022",
            "Prada2013",
            "Ramadass2004",
            "Xu2019",
        ]:
            out = name
        else:
            raise ValueError(f"Parameter set '{name}' not found")

        warnings.warn(
            f"Parameter sets should be called directly by their name ({name}),"
            f"instead of via pybamm.parameter_sets (pybamm.parameter_sets.{name}).",
            DeprecationWarning,
        )
        return out


parameter_sets = ParameterSets()
