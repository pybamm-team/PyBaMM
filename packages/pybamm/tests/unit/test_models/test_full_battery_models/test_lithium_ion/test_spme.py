#
# Tests for the lithium-ion SPMe model
#
import pytest

import pybamm
from tests import BaseUnitTestLithiumIon


class TestSPMe(BaseUnitTestLithiumIon):
    def setup_method(self):
        self.model = pybamm.lithium_ion.SPMe

    def test_electrolyte_options(self):
        options = {"electrolyte conductivity": "full"}
        with pytest.raises(pybamm.OptionError, match=r"electrolyte conductivity"):
            pybamm.lithium_ion.SPMe(options)

    def test_integrated_conductivity(self):
        options = {"electrolyte conductivity": "integrated"}
        self.check_well_posedness(options)

    def test_surface_form_c_e_av_is_variable(self):
        for surf in ["algebraic", "differential"]:
            model = pybamm.lithium_ion.SPMe(
                {"particle phases": ("2", "1"), "surface form": surf}
            )
            c_e_av = model.variables["X-averaged electrolyte concentration [mol.m-3]"]
            assert isinstance(c_e_av, pybamm.Variable)

    def test_no_surface_form_c_e_av_is_expression(self):
        model = pybamm.lithium_ion.SPMe()
        c_e_av = model.variables["X-averaged electrolyte concentration [mol.m-3]"]
        assert not isinstance(c_e_av, pybamm.Variable)

    def test_surface_form_macinnes_variable(self):
        for surf in ["algebraic", "differential"]:
            model = pybamm.lithium_ion.SPMe(
                {"particle phases": ("2", "1"), "surface form": surf}
            )
            assert "X-averaged negative MacInnes function" in model.variables
            macinnes = model.variables["X-averaged negative MacInnes function"]
            assert isinstance(macinnes, pybamm.Variable)

    def test_surface_form_phi_e_p_av_is_variable(self):
        for surf in ["algebraic", "differential"]:
            model = pybamm.lithium_ion.SPMe(
                {"particle phases": ("2", "1"), "surface form": surf}
            )
            key = "X-averaged positive electrolyte potential [V]"
            assert key in model.variables
            assert isinstance(model.variables[key], pybamm.Variable)

    def test_no_surface_form_phi_e_p_av_is_expression(self):
        model = pybamm.lithium_ion.SPMe()
        key = "X-averaged positive electrolyte potential [V]"
        assert key in model.variables
        assert not isinstance(model.variables[key], pybamm.Variable)

    def test_surface_form_sparsity_fixes_property(self):
        from pybamm.models.submodels.electrolyte_conductivity.composite_conductivity import (
            Composite,
        )

        param = pybamm.LithiumIonParameters()
        opts_surface = pybamm.BatteryModelOptions({"surface form": "algebraic"})
        opts_no_surface = pybamm.BatteryModelOptions({})

        sub = Composite(param, domain="negative", options=opts_surface)
        assert sub._use_surface_form_sparsity_fixes is True

        sub = Composite(param, domain="positive", options=opts_surface)
        assert sub._use_surface_form_sparsity_fixes is False

        sub = Composite(param, domain="negative", options=opts_no_surface)
        assert sub._use_surface_form_sparsity_fixes is False

        sub = Composite(param, domain=None, options=opts_surface)
        assert sub._use_surface_form_sparsity_fixes is True

    def test_surface_form_jacobian_sparsity(self):
        model = pybamm.lithium_ion.SPMe({"particle phases": ("2", "1")})
        sim = pybamm.Simulation(
            model,
            parameter_values=pybamm.ParameterValues("Chen2020_composite"),
            solver=pybamm.IDAKLUSolver(),
        )
        sim.build()
        sim._solver.set_up(sim._built_model)
        J = sim._solver.get_jacobian_sparsity()
        assert J.nnz < 900, f"Jacobian nnz={J.nnz} should be < 900 (was 3127)"
