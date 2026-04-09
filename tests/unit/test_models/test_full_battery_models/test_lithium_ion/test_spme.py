#
# Tests for the lithium-ion SPMe model
#
import pytest

import pybamm
from tests import BaseUnitTestLithiumIon
import casadi

    

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
            c_e_av = model.variables[
                "X-averaged electrolyte concentration [mol.m-3]"
            ]
            assert isinstance(c_e_av, pybamm.Variable)

    def test_no_surface_form_c_e_av_is_expression(self):
        model = pybamm.lithium_ion.SPMe()
        c_e_av = model.variables[
            "X-averaged electrolyte concentration [mol.m-3]"
        ]
        assert not isinstance(c_e_av, pybamm.Variable)

    def test_surface_form_macinnes_variable(self):
        for surf in ["algebraic", "differential"]:
            model = pybamm.lithium_ion.SPMe(
                {"particle phases": ("2", "1"), "surface form": surf}
            )
            assert "X-averaged negative MacInnes function" in model.variables
            macinnes = model.variables["X-averaged negative MacInnes function"]
            assert isinstance(macinnes, pybamm.Variable)

    def test_surface_form_sparsity_improvement(self):
        model = pybamm.lithium_ion.SPMe({"particle phases": ("2", "1")})
        pv = pybamm.ParameterValues("Chen2020_composite")
        sim = pybamm.Simulation(
            model, parameter_values=pv, solver=pybamm.IDAKLUSolver()
        )
        sim.solve([0, 100])
        dm = sim.built_model
        n = dm.len_rhs + dm.len_alg
        w = casadi.MX.sym("w", n)
        rhs = dm.concatenated_rhs.to_casadi(casadi.MX.sym("t"), w, inputs={})
        alg = dm.concatenated_algebraic.to_casadi(casadi.MX.sym("t"), w, inputs={})
        J = casadi.Function("J", [w], [casadi.jacobian(casadi.vertcat(rhs, alg), w)])
        nnz = J.sparsity_out(0).nnz()
        assert nnz < 1500, f"Jacobian nnz={nnz} should be < 1500 (was 3127 before fix)"
