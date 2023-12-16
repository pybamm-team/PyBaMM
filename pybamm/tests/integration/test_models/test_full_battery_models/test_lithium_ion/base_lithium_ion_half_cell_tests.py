#
# Base integration tests for lithium-ion half-cell battery models.
#
import tests
import pybamm


class BaseIntegrationTestLithiumIonHalfCell:
    def run_basic_processing_test(self, options, **kwargs):
        options["working electrode"] = "positive"
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, **kwargs)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing(self):
        options = {}
        self.run_basic_processing_test(options)

    def test_kinetics_asymmetric_butler_volmer(self):
        options = {"intercalation kinetics": "asymmetric Butler-Volmer"}
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Negative electrode Butler-Volmer transfer coefficient": 0.6,
                "Positive electrode Butler-Volmer transfer coefficient": 0.6,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_kinetics_linear(self):
        options = {"intercalation kinetics": "linear"}
        self.run_basic_processing_test(options)

    def test_kinetics_mhc(self):
        options = {"intercalation kinetics": "Marcus-Hush-Chidsey"}
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Negative electrode reorganization energy [eV]": 0.35,
                "Positive electrode reorganization energy [eV]": 0.35,
                "Positive electrode exchange-current density [A.m-2]": 5,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_irreversible_plating_with_porosity(self):
        options = {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
        }
        parameter_values = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameter_values.update({"Current function [A]": -2.5})  # C/2 charge
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_sei_constant(self):
        options = {"SEI": "constant"}
        self.run_basic_processing_test(options)

    def test_sei_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        self.run_basic_processing_test(options)

    def test_sei_asymmetric_reaction_limited(self):
        options = {"SEI": "reaction limited (asymmetric)"}
        parameter_values = pybamm.ParameterValues("Ecker2015_graphite_halfcell")
        parameter_values.update(
            {"SEI growth transfer coefficient": 0.2, "Current function [A]": -0.07826},
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_sei_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        self.run_basic_processing_test(options)

    def test_sei_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        self.run_basic_processing_test(options)

    def test_sei_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        self.run_basic_processing_test(options)

    def test_sei_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited"}
        self.run_basic_processing_test(options)

    def test_sei_asymmetric_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited (asymmetric)"}
        parameter_values = pybamm.ParameterValues("Ecker2015_graphite_halfcell")
        parameter_values.update(
            {"SEI growth transfer coefficient": 0.2, "Current function [A]": -0.07826},
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_swelling_only(self):
        options = {"particle mechanics": "swelling only"}
        parameter_values = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameter_values.update({"Current function [A]": -2.5})  # C/2 charge
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_constant_utilisation(self):
        options = {"interface utilisation": "constant"}
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Initial negative electrode interface utilisation": 0.9,
                "Initial positive electrode interface utilisation": 0.8,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_current_driven_utilisation(self):
        options = {"interface utilisation": "current-driven"}
        parameter_values = pybamm.ParameterValues("Xu2019")
        parameter_values.update(
            {
                "Initial negative electrode interface utilisation": 0.9,
                "Initial positive electrode interface utilisation": 0.8,
                "Negative electrode current-driven interface utilisation factor "
                "[m3.mol-1]": -1,
                "Positive electrode current-driven interface utilisation factor "
                "[m3.mol-1]": 1e-5,
            },
            check_already_exists=False,
        )
        self.run_basic_processing_test(options, parameter_values=parameter_values)

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        self.run_basic_processing_test(options)

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        self.run_basic_processing_test(options)
