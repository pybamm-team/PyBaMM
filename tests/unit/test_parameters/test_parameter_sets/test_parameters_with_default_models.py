#
# Tests each parameter set with the standard model associated with that parameter set
#

import pybamm


class TestParameterValuesWithModel:
    def test_parameter_values_with_model(self, subtests):
        param_to_model = {
            "Ai2020": pybamm.lithium_ion.DFN(
                {"particle mechanics": "swelling and cracking"}
            ),
            "Chen2020": pybamm.lithium_ion.DFN(),
            "Chen2020_composite": pybamm.lithium_ion.DFN(
                {
                    "particle phases": ("2", "1"),
                    "open-circuit potential": (("single", "current sigmoid"), "single"),
                }
            ),
            "Ecker2015": pybamm.lithium_ion.DFN(),
            "Ecker2015_graphite_halfcell": pybamm.lithium_ion.DFN(
                {"working electrode": "positive"}
            ),
            "Mohtat2020": pybamm.lithium_ion.DFN(),
            "NCA_Kim2011": pybamm.lithium_ion.DFN(),
            "OKane2022": pybamm.lithium_ion.DFN(
                {
                    "SEI": "solvent-diffusion limited",
                    "lithium plating": "partially reversible",
                }
            ),
            "OKane2022_graphite_SiOx_halfcell": pybamm.lithium_ion.DFN(
                {
                    "working electrode": "positive",
                    "SEI": "solvent-diffusion limited",
                    "lithium plating": "partially reversible",
                }
            ),
            "ORegan2022": pybamm.lithium_ion.DFN(),
            "Prada2013": pybamm.lithium_ion.DFN(),
            "Ramadass2004": pybamm.lithium_ion.DFN(),
            "Xu2019": pybamm.lithium_ion.DFN({"working electrode": "positive"}),
        }

        # Loop over each parameter set, testing that parameters can be set
        for param, model in param_to_model.items():
            with subtests.test(param=param):
                parameter_values = pybamm.ParameterValues(param)
                parameter_values.process_model(model)
