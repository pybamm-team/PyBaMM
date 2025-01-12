"""
Tests for the base_parameters.py
"""

import pybamm
import pytest


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

    def test__setattr__(self):
        # domain gets added as a subscript
        param = pybamm.GeometricParameters()
        assert param.n.L.print_name == r"L_{\mathrm{n}}"
