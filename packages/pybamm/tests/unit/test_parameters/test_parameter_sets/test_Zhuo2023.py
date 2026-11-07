#
# Tests for Zhuo (2023) parameter set
#
import pytest

import pybamm


class TestZhuo2023:
    def test_functions(self):
        param = pybamm.ParameterValues("Zhuo2023")
        sto = pybamm.Scalar(0.5)
        T = pybamm.Scalar(298.15)

        fun_test = {
            # Negative electrode (SiC)
            "Negative particle diffusivity [m2.s-1]": ([sto, T], 1e-14),
            "Negative electrode exchange-current density [A.m-2]": (
                [1000, 15000, 34257, T],
                0.5186,
            ),
            # Positive electrode (NMC811)
            "Positive particle diffusivity [m2.s-1]": ([sto, T], 1e-14),
            "Positive electrode exchange-current density [A.m-2]": (
                [1000, 20000, 49340, T],
                2.3651,
            ),
        }

        for name, value in fun_test.items():
            assert param.evaluate(param[name](*value[0])) == pytest.approx(
                value[1], rel=1e-4
            )

    def test_initial_oxygen_concentration(self):
        # Initial shell oxygen concentration is c_o_ini_ref * (1 - r_sh_nd)^2
        # with c_o_ini_ref = 15219.321
        param = pybamm.ParameterValues("Zhuo2023")
        f = param["Initial oxygen concentration in positive shell [mol.m-3]"]
        # At r_sh_nd = 0 (inner shell boundary): c_o = c_o_ini_ref
        r_sh_nd = pybamm.Scalar(0.0)
        x = pybamm.Scalar(1.0)
        assert param.evaluate(f(r_sh_nd, x)) == pytest.approx(15219.321, rel=1e-6)
        # At r_sh_nd = 1 (outer shell boundary): c_o = 0
        r_sh_nd = pybamm.Scalar(1.0)
        assert param.evaluate(f(r_sh_nd, x)) == pytest.approx(0.0, abs=1e-9)

    def test_pe_degradation_parameter_values(self):
        # Verify the key PE phase-transition parameters have the documented values
        param = pybamm.ParameterValues("Zhuo2023")
        assert param["Positive shell oxygen diffusivity [m2.s-1]"] == 1e-17
        assert param["Forward chemical reaction coefficient [m.s-1]"] == 0.8544e-11
        assert (
            param["Reverse chemical reaction coefficient [m4.mol-1.s-1]"] == 1.732e-16
        )
        assert param["Trapped lithium concentration in the shell [mol.m-3]"] == 20000
        assert (
            param["Threshold lithium concentration for phase transition [mol.m-3]"]
            == 14802
        )
        assert param["Positive electrode shell resistivity [Ohm.m]"] == 1e6
        assert param["Constant oxygen concentration in the core [mol.m-3]"] == 152193.21
        assert param["Initial core-shell phase boundary location"] == 0.9868421

    def test_citations(self):
        param = pybamm.ParameterValues("Zhuo2023")
        assert "Zhuo2023" in param["citations"]

    def test_process_model_with_phase_transition_dfn(self):
        # SPM coverage lives in test_parameters_with_default_models.py; this
        # adds the DFN counterpart that exercises the base_kinetics PE-shell
        # branch rather than the inverse Butler-Volmer hook.
        model = pybamm.lithium_ion.DFN({"PE degradation": "phase transition"})
        param = pybamm.ParameterValues("Zhuo2023")
        param.process_model(model)
        model.check_well_posedness()
