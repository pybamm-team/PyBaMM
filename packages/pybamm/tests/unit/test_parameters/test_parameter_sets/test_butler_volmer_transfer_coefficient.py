#
# Tests that lithium-ion parameter sets provide the Butler-Volmer transfer coefficient
# read by asymmetric Butler-Volmer kinetics
#
import pytest

import pybamm

HALF_CELL_SETS = [
    "Ecker2015_graphite_halfcell",
    "OKane2022_graphite_SiOx_halfcell",
    "Xu2019",
]
COMPOSITE_SETS = ["Bonkile2024", "Chen2020_composite"]
PARAMETER_SETS = [
    "Ai2020",
    "Chen2020",
    "Marquis2019",
    "Mohtat2020",
    "NCA_Kim2011",
    "OKane2022",
    "ORegan2022",
    "Prada2013",
    "Ramadass2004",
    *HALF_CELL_SETS,
    *COMPOSITE_SETS,
]


class TestButlerVolmerTransferCoefficient:
    @pytest.mark.parametrize("parameter_set", PARAMETER_SETS)
    def test_parameter_set_supports_asymmetric_butler_volmer(self, parameter_set):
        parameter_values = pybamm.ParameterValues(parameter_set)
        for domain in ["Negative", "Positive"]:
            assert f"{domain} electrode charge transfer coefficient" not in (
                parameter_values
            )

        options = {"intercalation kinetics": "asymmetric Butler-Volmer"}
        if parameter_set in HALF_CELL_SETS:
            options["working electrode"] = "positive"
        if parameter_set in COMPOSITE_SETS:
            options["particle phases"] = ("2", "1")
        parameter_values.process_model(pybamm.lithium_ion.SPM(options))
