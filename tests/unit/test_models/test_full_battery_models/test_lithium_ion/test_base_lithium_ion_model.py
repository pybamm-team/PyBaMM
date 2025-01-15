#
# Tests for the base lead acid model class
#
import pybamm
import os
import pytest


class TestBaseLithiumIonModel:
    def test_incompatible_options(self):
        with pytest.raises(pybamm.OptionError, match="convection not implemented"):
            pybamm.lithium_ion.BaseModel({"convection": "uniform transverse"})

    def test_default_parameters(self):
        # check parameters are read in ok
        model = pybamm.lithium_ion.BaseModel()
        assert model.default_parameter_values["Reference temperature [K]"] == 298.15

        # change path and try again

        cwd = os.getcwd()
        os.chdir("..")
        model = pybamm.lithium_ion.BaseModel()
        assert model.default_parameter_values["Reference temperature [K]"] == 298.15
        os.chdir(cwd)

    def test_insert_reference_electrode(self):
        model = pybamm.lithium_ion.SPM()
        model.insert_reference_electrode()
        assert "Negative electrode 3E potential [V]" in model.variables
        assert "Positive electrode 3E potential [V]" in model.variables
        assert "Reference electrode potential [V]" in model.variables

        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        model.insert_reference_electrode()
        assert "Negative electrode potential [V]" not in model.variables
        assert "Positive electrode 3E potential [V]" in model.variables
        assert "Reference electrode potential [V]" in model.variables

        model = pybamm.lithium_ion.SPM({"dimensionality": 2})
        with pytest.raises(
            NotImplementedError, match="Reference electrode can only be"
        ):
            model.insert_reference_electrode()
