#
# Base integration tests for lithium-ion half-cell battery models.
#
import tests


class BaseIntegrationTestLithiumIonHalfCell:
    def run_basic_processing_test(self, options, **kwargs):
        options["working electrode"] = "positive"
        model = self.model(options)
        modeltest = tests.StandardModelTest(model, **kwargs)
        modeltest.test_all(skip_output_tests=True)

    def test_basic_processing(self):
        options = {}
        self.run_basic_processing_test(options)

    def test_constant_sei(self):
        options = {"SEI": "constant"}
        self.run_basic_processing_test(options)

    def test_reaction_limited_sei(self):
        options = {"SEI": "reaction limited"}
        self.run_basic_processing_test(options)

    def test_solvent_diffusion_limited_sei(self):
        options = {"SEI": "solvent-diffusion limited"}
        self.run_basic_processing_test(options)

    def test_electron_migration_limited_sei(self):
        options = {"SEI": "electron-migration limited"}
        self.run_basic_processing_test(options)

    def test_interstitial_diffusion_limited_sei(self):
        options = {"SEI": "interstitial-diffusion limited"}
        self.run_basic_processing_test(options)

    def test_ec_reaction_limited_sei(self):
        options = {"SEI": "ec reaction limited"}
        self.run_basic_processing_test(options)
