#
# Tests for the lithium-ion SPMe model
#
import numpy as np
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

    @pytest.mark.parametrize(
        ("surface_form", "working_electrode"),
        [
            ("algebraic", "both"),
            ("differential", "both"),
            ("false", "positive"),
            ("algebraic", "positive"),
            ("differential", "positive"),
        ],
    )
    def test_integrated_conductivity_unsupported_raises(
        self, surface_form, working_electrode
    ):
        options = {
            "electrolyte conductivity": "integrated",
            "surface form": surface_form,
            "working electrode": working_electrode,
        }
        with pytest.raises(pybamm.OptionError, match=r"'integrated' is only supported"):
            pybamm.lithium_ion.SPMe(options)

    @pytest.mark.parametrize(
        ("options", "parameter_set"),
        [
            ({"surface form": "algebraic"}, None),
            ({"surface form": "differential"}, None),
            (
                {
                    "open-circuit potential": "MSMR",
                    "particle": "MSMR",
                    "number of MSMR reactions": ("6", "4"),
                    "intercalation kinetics": "MSMR",
                    "surface form": "algebraic",
                },
                "MSMR_Example",
            ),
        ],
    )
    def test_surface_form_reaction_is_uniform(self, options, parameter_set):
        model = pybamm.lithium_ion.SPMe(options)
        parameter_values = (
            pybamm.ParameterValues(parameter_set)
            if parameter_set
            else model.default_parameter_values
        )
        sim = pybamm.Simulation(model, parameter_values=parameter_values, C_rate=2)
        solution = sim.solve([0, 600])
        t = 600
        x_n = sim.mesh["negative electrode"].nodes

        a_j = solution[
            "Negative electrode volumetric interfacial current density [A.m-3]"
        ](t, x=x_n)
        i_cc = solution["Current collector current density [A.m-2]"](t)
        L_n = parameter_values["Negative electrode thickness [m]"]
        np.testing.assert_allclose(a_j, a_j[0], rtol=1e-12)
        # loose enough for the default solver tolerances and the double layer
        np.testing.assert_allclose(a_j, i_cc / L_n, rtol=1e-3)

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

    def test_c_e_av_at_rest_equals_initial_concentration(self):
        for surf in ["false", "algebraic", "differential"]:
            model = pybamm.lithium_ion.SPMe({"surface form": surf})
            parameter_values = model.default_parameter_values
            parameter_values["Current function [A]"] = 0
            sim = pybamm.Simulation(model, parameter_values=parameter_values)
            solution = sim.solve([0, 10])
            c_e_av = solution["X-averaged electrolyte concentration [mol.m-3]"]
            np.testing.assert_allclose(
                c_e_av(solution.t),
                parameter_values["Initial concentration in electrolyte [mol.m-3]"],
                rtol=1e-8,
            )

    @pytest.mark.parametrize("surface_form", ["algebraic", "differential"])
    def test_surface_form_initial_conditions_consistent_at_rest(self, surface_form):
        model = pybamm.lithium_ion.SPMe({"surface form": surface_form})
        parameter_values = model.default_parameter_values
        parameter_values["Current function [A]"] = 0
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        solution = sim.solve([0, 10])
        y_guess = sim.built_model.concatenated_initial_conditions.evaluate(0)
        np.testing.assert_allclose(
            solution.all_ys[0][:, 0], y_guess.ravel(), rtol=1e-6, atol=1e-8
        )

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
