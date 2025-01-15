#
# Tests for the lithium-ion SPM model
#
import pybamm
from tests import BaseUnitTestLithiumIon
import pytest


class TestSPM(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPM

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with pytest.raises(pybamm.OptionError, match="electrolyte conductivity"):
            pybamm.lithium_ion.SPM(options)

    def test_kinetics_options(self):
        options = {
            "surface form": "false",
            "intercalation kinetics": "Marcus-Hush-Chidsey",
        }
        with pytest.raises(pybamm.OptionError, match="Inverse kinetics"):
            pybamm.lithium_ion.SPM(options)

    def test_x_average_options(self):
        # Check model with x-averaged side reactions
        options = {
            "lithium plating": "irreversible",
            "lithium plating porosity change": "true",
            "SEI": "ec reaction limited",
            "SEI porosity change": "true",
            "x-average side reactions": "true",
        }
        self.check_well_posedness(options)

        # Check model with distributed side reactions throws an error
        options["x-average side reactions"] = "false"
        with pytest.raises(pybamm.OptionError, match="cannot be 'false' for SPM"):
            pybamm.lithium_ion.SPM(options)

    def test_distribution_options(self):
        with pytest.raises(pybamm.OptionError, match="surface form"):
            pybamm.lithium_ion.SPM({"particle size": "distribution"})

    def test_particle_size_distribution(self):
        options = {"surface form": "algebraic", "particle size": "distribution"}
        self.check_well_posedness(options)

    def test_new_model(self):
        model = pybamm.lithium_ion.SPM({"thermal": "x-full"})
        new_model = model.new_copy()
        model_T_eqn = model.rhs[model.variables["Cell temperature [K]"]]
        new_model_T_eqn = new_model.rhs[new_model.variables["Cell temperature [K]"]]
        assert new_model_T_eqn == model_T_eqn
        assert new_model.name == model.name
        assert new_model.use_jacobian == model.use_jacobian
        assert new_model.convert_to_format == model.convert_to_format

        # with custom submodels
        options = {"stress-induced diffusion": "false", "thermal": "x-full"}
        model = pybamm.lithium_ion.SPM(options, build=False)
        particle_n = pybamm.particle.XAveragedPolynomialProfile(
            model.param,
            "negative",
            {**options, "particle": "quadratic profile"},
            "primary",
        )
        model.submodels["negative primary particle"] = particle_n
        model.build_model()
        new_model = model.new_copy()
        new_model_cs_eqn = list(new_model.rhs.values())[1]
        model_cs_eqn = list(model.rhs.values())[1]
        assert new_model_cs_eqn == model_cs_eqn
