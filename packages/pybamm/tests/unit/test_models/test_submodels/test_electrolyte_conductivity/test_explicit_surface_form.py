import pytest

import pybamm


class TestExplicitSurfaceForm:
    @pytest.mark.parametrize("domain", ["negative", "positive"])
    @pytest.mark.parametrize("inverse_kinetics", [False, True])
    def test_x_averaged_variable(self, domain, inverse_kinetics):
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.electrolyte_conductivity.surface_potential_form.Explicit(
            param, domain, {}, inverse_kinetics=inverse_kinetics
        )
        submodel.build()
        Domain = domain.capitalize()
        assert (
            f"{Domain} electrode surface potential difference [V]" in submodel.variables
        )
        name = f"X-averaged {domain} electrode surface potential difference [V]"
        assert (name in submodel.variables) is not inverse_kinetics

    def test_separator_defines_nothing(self):
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.electrolyte_conductivity.surface_potential_form.Explicit(
            param, "separator", {}
        )
        submodel.build()
        assert submodel.variables == {}

    def test_spm_takes_x_average_from_inverse_kinetics(self):
        model = pybamm.lithium_ion.SPM()
        submodel = model.submodels["negative surface potential difference"]
        assert submodel.inverse_kinetics
        assert (
            "X-averaged negative electrode surface potential difference [V]"
            in model.variables
        )


class TestIntegratedConductivity:
    def test_planar_negative_electrode_raises(self):
        with pytest.raises(pybamm.ModelError, match=r"porous negative electrode"):
            pybamm.lithium_ion.SPMe(
                {
                    "electrolyte conductivity": "integrated",
                    "working electrode": "positive",
                }
            )
