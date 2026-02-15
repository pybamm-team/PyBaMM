#
# Tests for the lithium-ion SPM model
#
import pytest

import pybamm
from tests import BaseUnitTestLithiumIon


class TestSPM(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPM

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with pytest.raises(pybamm.OptionError, match=r"electrolyte conductivity"):
            pybamm.lithium_ion.SPM(options)

    def test_kinetics_options(self):
        options = {
            "surface form": "false",
            "intercalation kinetics": "Marcus-Hush-Chidsey",
        }
        with pytest.raises(pybamm.OptionError, match=r"Inverse kinetics"):
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
        with pytest.raises(pybamm.OptionError, match=r"cannot be 'false' for SPM"):
            pybamm.lithium_ion.SPM(options)

    def test_distribution_options(self):
        with pytest.raises(pybamm.OptionError, match=r"surface form"):
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

    def test_basic_spm_with_3d_thermal_pouch(self):
        options = {"cell geometry": "pouch", "dimensionality": 3}
        self.model = pybamm.lithium_ion.Basic3DThermalSPM
        self.check_well_posedness(options)

    def test_basic_spm_with_3d_thermal_cylinder(self):
        options = {"cell geometry": "cylindrical", "dimensionality": 3}
        self.model = pybamm.lithium_ion.Basic3DThermalSPM
        self.check_well_posedness(options)

    def test_basic_spm_with_3d_thermal_incompatible_options(self):
        options = {"cell geometry": "cylindrical", "dimensionality": 2}
        self.model = pybamm.lithium_ion.Basic3DThermalSPM
        with pytest.raises(
            pybamm.OptionError,
            match=r"'dimensionality' must be '3' if 'cell geometry' is 'cylindrical'",
        ):
            self.check_well_posedness(options)

        options = {"cell geometry": "arbitrary", "dimensionality": 3}
        with pytest.raises(
            pybamm.OptionError,
            match=r"'cell geometry' must be 'pouch' or 'cylindrical' if 'dimensionality' is '3'",
        ):
            self.check_well_posedness(options)
