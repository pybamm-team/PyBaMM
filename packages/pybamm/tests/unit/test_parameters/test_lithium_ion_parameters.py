#
# Tests lithium-ion parameters load and give expected values
#

import numpy as np
import pytest

import pybamm
from pybamm.parameters.lithium_ion_parameters import (
    U_asymptote_approaching_zero,
    U_asymptotes,
)


class TestLithiumIonParameterValues:
    def test_print_parameters(self, tmp_path):
        parameters = pybamm.LithiumIonParameters()
        parameter_values = pybamm.lithium_ion.BaseModel().default_parameter_values
        output_file = tmp_path / "lithium_ion_parameters.txt"
        parameter_values.print_parameters(parameters, output_file)

    def test_lithium_ion(self):
        """This test checks that all the parameters are being calculated
        correctly for the specific set of parameters for LCO from dualfoil. The values
        are those converted from those in Scott's transfer which previous versions of
        the DFN work with. A 1C rate corresponds to a 24A/m^2 current density"""
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()

        # particle geometry
        np.testing.assert_almost_equal(
            values.evaluate(param.n.prim.a_typ), 0.18 * 10 ** (6), 2
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.n.prim.R_typ), 1 * 10 ** (-5), 2
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.p.prim.a_typ), 0.15 * 10 ** (6), 2
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.p.prim.R_typ), 1 * 10 ** (-5), 2
        )

        # j0_m
        np.testing.assert_almost_equal(
            values.evaluate(
                param.n.prim.j0(param.c_e_init_av, param.n.prim.c_max / 2, param.T_ref)
            ),
            values.evaluate(
                2 * 10 ** (-5) * param.c_e_init_av**0.5 * param.n.prim.c_max / 2
            ),
            8,
        )

        # j0_p
        np.testing.assert_almost_equal(
            values.evaluate(
                param.p.prim.j0(param.c_e_init_av, param.p.prim.c_max / 2, param.T_ref)
            ),
            values.evaluate(
                6 * 10 ** (-7) * param.c_e_init_av**0.5 * param.p.prim.c_max / 2
            ),
            8,
        )

        # particle dynamics
        # neg diffusion coefficient
        np.testing.assert_almost_equal(
            values.evaluate(
                pybamm.xyz_average(
                    pybamm.r_average(param.n.prim.D(param.n.prim.c_init, param.T_ref))
                )
            ),
            3.9 * 10 ** (-14),
            2,
        )

        # pos diffusion coefficient
        np.testing.assert_almost_equal(
            values.evaluate(
                pybamm.xyz_average(
                    pybamm.r_average(param.p.prim.D(param.p.prim.c_init, param.T_ref))
                )
            ),
            1 * 10 ** (-13),
            2,
        )

        # electrolyte dynamics
        np.testing.assert_almost_equal(
            values.evaluate(param.D_e(param.c_e_init_av, param.T_ref)),
            5.34 * 10 ** (-10) * np.exp(-0.65),
            10,
        )

        # electrolyte conductivity
        np.testing.assert_almost_equal(
            values.evaluate(param.kappa_e(param.c_e_init_av, param.T_ref)), 1.1045, 3
        )

        # electrode conductivities
        # neg
        np.testing.assert_almost_equal(
            values.evaluate(param.n.sigma(None, param.T_ref)), 100, 3
        )

        # pos
        np.testing.assert_almost_equal(
            values.evaluate(param.p.sigma(None, param.T_ref)), 10, 3
        )

    def test_thermal_parameters(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        values.update(
            {
                "Cell heat capacity [J.K-1.m-3]": 2.5e6,
                "Left face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Right face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Front face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Back face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Top face heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Inner radius heat transfer coefficient [W.m-2.K-1]": 5.0,
                "Outer radius heat transfer coefficient [W.m-2.K-1]": 5.0,
            }
        )
        param = pybamm.LithiumIonParameters()
        T = param.T_ref

        # Density
        np.testing.assert_equal(values.evaluate(param.n.rho_c_p_cc(T)), 8954 * 385)
        np.testing.assert_equal(values.evaluate(param.n.rho_c_p(T)), 1657 * 700)
        np.testing.assert_equal(values.evaluate(param.s.rho_c_p(T)), 397 * 700)
        np.testing.assert_equal(values.evaluate(param.p.rho_c_p(T)), 3262 * 700)
        np.testing.assert_equal(values.evaluate(param.p.rho_c_p_cc(T)), 2707 * 897)

        # Thermal conductivity
        np.testing.assert_equal(values.evaluate(param.n.lambda_cc(T)), 401)
        np.testing.assert_equal(values.evaluate(param.n.lambda_(T)), 1.7)
        np.testing.assert_equal(values.evaluate(param.s.lambda_(T)), 0.16)
        np.testing.assert_equal(values.evaluate(param.p.lambda_(T)), 2.1)
        np.testing.assert_equal(values.evaluate(param.p.lambda_cc(T)), 237)

        # other thermal parameters
        np.testing.assert_equal(values.evaluate(param.T_init), 298.15)
        np.testing.assert_equal(values.evaluate(param.cell_heat_capacity), 2.5e6)
        np.testing.assert_equal(values.evaluate(param.h_edge_x_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_x_max), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_y_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_y_max), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_z_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_z_max), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_radial_min), 5.0)
        np.testing.assert_equal(values.evaluate(param.h_edge_radial_max), 5.0)

    def test_parameter_functions(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()

        T_test = 298.15

        c_e_test = 1000
        values.evaluate(param.D_e(c_e_test, T_test))
        values.evaluate(param.kappa_e(c_e_test, T_test))

    def test_sigma_as_function_of_stoichiometry(self):
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()
        T = param.T_ref

        # a constant conductivity ignores the stoichiometry input, so passing sto
        # gives the same value as the temperature-only call (backwards compatible)
        np.testing.assert_almost_equal(
            values.evaluate(param.n.sigma(pybamm.Scalar(0.5), T)), 100, 3
        )

        # conductivity supplied as a function of stoichiometry and temperature
        values.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    100 * (1 + sto)
                ),
                "Positive electrode conductivity [S.m-1]": lambda sto, T: (
                    10 * (1 + sto)
                ),
            }
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.n.sigma(pybamm.Scalar(0.5), T)), 150, 3
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.p.sigma(pybamm.Scalar(0.2), T)), 12, 3
        )

        # stoichiometry is clipped into (tol, 1 - tol) so the function is never
        # evaluated at exactly 0 or 1
        tol = pybamm.settings.tolerances["sigma__c_s"]
        np.testing.assert_almost_equal(
            values.evaluate(param.n.sigma(pybamm.Scalar(5.0), T)),
            100 * (1 + (1 - tol)),
            3,
        )
        np.testing.assert_almost_equal(
            values.evaluate(param.n.sigma(pybamm.Scalar(-5.0), T)),
            100 * (1 + tol),
            3,
        )

    def test_sigma_temperature_only_function_raises(self):
        # a conductivity supplied as a function of temperature only (the old
        # signature) must raise a clear error pointing at f(sto, T), rather than
        # failing later with a cryptic "takes 1 positional argument" TypeError
        values = pybamm.lithium_ion.BaseModel().default_parameter_values
        param = pybamm.LithiumIonParameters()
        T = param.T_ref

        values.update({"Negative electrode conductivity [S.m-1]": lambda T: 100.0})
        with pytest.raises(TypeError, match=r"\(stoichiometry, temperature\)"):
            values.process_symbol(param.n.sigma(pybamm.Scalar(0.5), T))

        # a constant (scalar) conductivity is unaffected by the check
        values.update({"Negative electrode conductivity [S.m-1]": 100.0})
        values.process_symbol(param.n.sigma(pybamm.Scalar(0.5), T))


class TestUAsymptotes:
    """Tests for the OCP asymptote functions."""

    def test_U_asymptote_approaching_zero_values(self):
        """Test that U_asymptote_approaching_zero returns expected values."""
        # Test at sto = 0: should be ~1000 mV (1 V)
        val_at_zero = U_asymptote_approaching_zero(0.0).evaluate()
        assert val_at_zero == pytest.approx(1.0, rel=1e-3)

        # Test at sto = 0.001: should be ~1 mV
        val_at_001 = U_asymptote_approaching_zero(0.001).evaluate()
        assert val_at_001 == pytest.approx(0.001, rel=1e-2)

        # Test at sto = 1: should be essentially 0
        val_at_one = U_asymptote_approaching_zero(1.0).evaluate()
        assert val_at_one < 1e-10

    def test_U_asymptotes_antisymmetry(self):
        """Test that U_asymptotes is antisymmetric: U(sto) = -U(1-sto)."""
        test_points = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]

        for sto in test_points:
            u_sto = U_asymptotes(sto).evaluate()
            u_1_minus_sto = U_asymptotes(1 - sto).evaluate()
            # U(sto) + U(1-sto) should equal 0 (antisymmetry)
            assert u_sto + u_1_minus_sto == pytest.approx(0.0, abs=1e-12)

    def test_U_asymptotes_boundary_values(self):
        """Test U_asymptotes at boundary values."""

        # At sto = 0: should be positive (~1 V)
        assert U_asymptotes(0.0).evaluate() > 0
        assert U_asymptotes(0.0).evaluate() == pytest.approx(1.0, rel=1e-3)

        # At sto = 1: should be negative (~-1 V)
        assert U_asymptotes(1.0).evaluate() < 0
        assert U_asymptotes(1.0).evaluate() == pytest.approx(-1.0, rel=1e-3)

        # At sto = 0.5: should be 0
        assert U_asymptotes(0.5).evaluate() == pytest.approx(0.0, abs=1e-10)

    def test_U_asymptote_numerical_stability(self):
        """Test that U_asymptote_approaching_zero doesn't overflow for extreme values."""
        # For very negative stoichiometries, should return finite large values
        # This tests the logaddexp fix: np.log(1 + exp(7000)) would overflow
        val_neg = U_asymptote_approaching_zero(-1.0).evaluate()
        assert np.isfinite(val_neg)
        assert val_neg > 0  # Should be a large positive barrier

        val_very_neg = U_asymptote_approaching_zero(-10.0).evaluate()
        assert np.isfinite(val_very_neg)
        assert val_very_neg > val_neg  # More negative sto = larger barrier


class TestUInverse:
    """
    ``U_inverse`` covers provably monotonic tabulated potentials and refuses the rest.
    Of the shipped sets only Ai2020 and Chayambuka2022 have one, on the negative
    electrode; every other potential must raise rather than be attempted.
    """

    PROVABLE = [("Ai2020", "n"), ("Chayambuka2022", "n")]
    REFUSED = [
        ("Ai2020", "p"),  # tabulated, but the spline rings, so not monotonic
        ("Chayambuka2022", "p"),  # tabulated, data itself is not monotonic
        ("Chen2020", "n"),  # analytic
        ("Chen2020", "p"),  # analytic
        ("Ramadass2004", "p"),  # analytic, and a pole at sto ~ 0.2453
        ("OKane2022", "n"),  # tabulated, 117 monotonic stretches
    ]

    def _phase(self, param, electrode):
        return param.n.prim if electrode == "n" else param.p.prim

    @pytest.mark.parametrize(("parameter_set", "electrode"), REFUSED)
    def test_refuses_what_it_cannot_prove(self, parameter_set, electrode):
        parameter_values = pybamm.ParameterValues(parameter_set)
        phase = self._phase(pybamm.LithiumIonParameters(), electrode)
        with pytest.raises(pybamm.OptionError, match="cannot be shown to decrease"):
            phase.U_inverse(pybamm.Scalar(0.1), pybamm.Scalar(298.15), parameter_values)

    @pytest.mark.parametrize(("parameter_set", "electrode"), PROVABLE)
    def test_round_trips_through_U(self, parameter_set, electrode):
        import casadi

        parameter_values = pybamm.ParameterValues(parameter_set)
        phase = self._phase(pybamm.LithiumIonParameters(), electrode)
        symbol = casadi.MX.sym("value")
        forward = casadi.Function(
            "U",
            [symbol],
            [
                parameter_values.process_symbol(
                    phase.U(pybamm.InputParameter("s"), pybamm.Scalar(298.15))
                ).to_casadi(inputs={"s": symbol})
            ],
        )
        inverse = casadi.Function(
            "U_inverse",
            [symbol],
            [
                parameter_values.process_symbol(
                    phase.U_inverse(
                        pybamm.InputParameter("target"),
                        pybamm.Scalar(298.15),
                        parameter_values,
                    )
                ).to_casadi(inputs={"target": symbol})
            ],
        )
        for stoichiometry in np.linspace(0.05, 0.95, 25):
            target = float(forward(stoichiometry))
            got = float(inverse(target))
            np.testing.assert_allclose(got, stoichiometry, atol=1e-9)
            np.testing.assert_allclose(float(forward(got)), target, atol=1e-9)

    def test_the_asymptotes_make_any_target_reachable(self):
        import casadi

        # far outside the potential's own range, so only the barrier can meet it
        parameter_values = pybamm.ParameterValues("Ai2020")
        phase = pybamm.LithiumIonParameters().n.prim
        symbol = casadi.MX.sym("target")
        inverse = casadi.Function(
            "U_inverse",
            [symbol],
            [
                parameter_values.process_symbol(
                    phase.U_inverse(
                        pybamm.InputParameter("target"),
                        pybamm.Scalar(298.15),
                        parameter_values,
                    )
                ).to_casadi(inputs={"target": symbol})
            ],
        )
        for target in (50.0, -50.0):
            assert np.isfinite(float(inverse(target)))

    def test_bounds_are_only_a_starting_scale(self):
        import casadi

        parameter_values = pybamm.ParameterValues("Ai2020")
        phase = pybamm.LithiumIonParameters().n.prim
        symbol = casadi.MX.sym("target")

        def solve(target, **kwargs):
            node = phase.U_inverse(
                pybamm.InputParameter("target"),
                pybamm.Scalar(298.15),
                parameter_values,
                **kwargs,
            )
            function = casadi.Function(
                "f",
                [symbol],
                [
                    parameter_values.process_symbol(node).to_casadi(
                        inputs={"target": symbol}
                    )
                ],
            )
            return float(function(target))

        target = 0.1
        np.testing.assert_allclose(
            solve(target, sto_bounds=(0.01, 0.02)), solve(target), atol=1e-9
        )
        with pytest.raises(RuntimeError, match="rootfinder process failed"):
            solve(target, sto_bounds=(0.01, 0.02), max_expansions=0)


class TestUIsStrictlyDecreasing:
    def _processed(self, parameter_set):
        parameter_values = pybamm.ParameterValues(parameter_set)
        param = pybamm.LithiumIonParameters()
        temperature = pybamm.Scalar(parameter_values.evaluate(param.T_ref))
        return [
            parameter_values.process_symbol(
                phase.U(pybamm.Variable("sto"), temperature)
            )
            for phase in (param.n.prim, param.p.prim)
        ]

    def test_the_barrier_free_window_is_where_the_barrier_vanishes(self):
        from pybamm.parameters.lithium_ion_parameters import (
            U_BARRIER_FREE,
            U_asymptotes,
        )

        # strictly inside, the barrier is bit-exactly zero, so it contributes nothing
        # to U. The closed form itself is the last point before that, within an ulp.
        inside = np.linspace(U_BARRIER_FREE, 1 - U_BARRIER_FREE, 1001)[1:-1]
        assert all(
            float(U_asymptotes(pybamm.Scalar(s)).evaluate()) == 0.0 for s in inside
        )
        assert float(U_asymptotes(pybamm.Scalar(U_BARRIER_FREE)).evaluate()) < 1e-13
        # and well outside it does not vanish at all
        assert float(U_asymptotes(pybamm.Scalar(0.0)).evaluate()) == pytest.approx(1.0)

    def test_the_barrier_never_increases(self):
        # what the proof leans on: U' = interpolant' + barrier', and the second term is
        # never positive anywhere, so it can only reinforce a decreasing interpolant
        from pybamm.parameters.lithium_ion_parameters import U_asymptotes

        sto = pybamm.StateVector(slice(0, 1))
        derivative = U_asymptotes(sto).diff(sto)
        for point in (-2.0, -0.01, 0.0, 1e-4, 0.001, 0.5, 0.999, 1.0, 3.0):
            value = float(np.asarray(derivative.evaluate(y=np.array([point]))).item())
            assert value <= 0.0, (point, value)

    @pytest.mark.parametrize(
        ("parameter_set", "expected"),
        [
            # tabulated and decreasing over the window, so provable
            ("Ai2020", [True, False]),
            ("Chayambuka2022", [True, False]),
            # analytic: nothing to read, so never provable
            ("Chen2020", [False, False]),
            ("Ramadass2004", [False, False]),
        ],
    )
    def test_proves_only_what_it_can(self, parameter_set, expected):
        from pybamm.parameters.lithium_ion_parameters import (
            U_BARRIER_FREE,
            U_is_strictly_decreasing,
        )

        region = (U_BARRIER_FREE, 1 - U_BARRIER_FREE)
        got = [
            U_is_strictly_decreasing(u, region) for u in self._processed(parameter_set)
        ]
        assert got == expected

    def test_an_increasing_potential_is_refused(self):
        from pybamm.parameters.lithium_ion_parameters import U_is_strictly_decreasing

        sto = pybamm.Variable("sto")
        x = np.linspace(0, 1, 11)
        rising = pybamm.Interpolant(x, x, sto, interpolator="linear")
        assert U_is_strictly_decreasing(rising + pybamm.Scalar(1), (0.1, 0.9)) is False
        # entering by subtraction is not the shape the proof covers
        falling = pybamm.Interpolant(x, -x, sto, interpolator="linear")
        assert U_is_strictly_decreasing(falling + pybamm.Scalar(1), (0.1, 0.9)) is True
        assert U_is_strictly_decreasing(pybamm.Scalar(1) - falling, (0.1, 0.9)) is False
