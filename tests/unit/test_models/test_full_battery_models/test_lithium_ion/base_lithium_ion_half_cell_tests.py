#
# Base unit tests for lithium-ion half-cell models
# This is achieved by using the {"working electrdode": "positive"} option
#


class BaseUnitTestLithiumIonHalfCell:
    def check_well_posedness(self, options):
        if self.model is not None:
            options["working electrode"] = "positive"
            model = self.model(options)
            model.check_well_posedness()

    def test_well_posed(self):
        options = {}
        self.check_well_posedness(options)

    def test_well_posed_constant(self):
        options = {"SEI": "constant"}
        self.check_well_posedness(options)

    def test_well_posed_reaction_limited(self):
        options = {"SEI": "reaction limited"}
        self.check_well_posedness(options)

    def test_well_posed_solvent_diffusion_limited(self):
        options = {"SEI": "solvent-diffusion limited"}
        self.check_well_posedness(options)

    def test_well_posed_electron_migration_limited(self):
        options = {"SEI": "electron-migration limited"}
        self.check_well_posedness(options)

    def test_well_posed_interstitial_diffusion_limited(self):
        options = {"SEI": "interstitial-diffusion limited"}
        self.check_well_posedness(options)

    def test_well_posed_ec_reaction_limited(self):
        options = {"SEI": "ec reaction limited"}
        self.check_well_posedness(options)
