"""
Tests for the base_parameters.py
"""

import pytest

import pybamm


class TestBaseParameters:
    def test_getattr__(self):
        param = pybamm.LithiumIonParameters()
        # ending in _n / _s / _p
        with pytest.raises(AttributeError, match="param.n.L"):
            param.L_n
        with pytest.raises(AttributeError, match="param.s.L"):
            param.L_s
        with pytest.raises(AttributeError, match="param.p.L"):
            param.L_p
        # _n_ in the name
        with pytest.raises(AttributeError, match="param.n.prim.c_max"):
            param.c_n_max

        # _n_ or _p_ not in name
        with pytest.raises(
            AttributeError, match="has no attribute 'c_n_not_a_parameter"
        ):
            param.c_n_not_a_parameter

        with pytest.raises(AttributeError, match="has no attribute 'c_s_test"):
            pybamm.electrical_parameters.c_s_test

        assert param.n.cap_init == param.n.Q_init
        assert param.p.prim.cap_init == param.p.prim.Q_init

    def test_that_initial_state_function_is_assigned(self):
        param_1 = pybamm.ParameterValues("Chen2020")
        assert param_1._set_initial_state == pybamm.lithium_ion.set_initial_state

        def my_func(x):
            return x

        param_1.update({"Initial state function": my_func})
        assert param_1._set_initial_state == my_func
        param_2 = pybamm.ParameterValues("ECM_Example")
        assert param_2._set_initial_state == pybamm.equivalent_circuit.set_initial_state

        def my_error_func(*args, **kwargs):
            raise NotImplementedError("this function should error")

        param_1.update({"Initial state function": my_error_func})
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model, parameter_values=param_1)
        with pytest.raises(NotImplementedError, match="this function should error"):
            sim.solve([0, 1], initial_soc=0.5)

    def test__setattr__(self):
        # domain gets added as a subscript
        param = pybamm.GeometricParameters()
        assert param.n.L.print_name == r"L_{\mathrm{n}}"
