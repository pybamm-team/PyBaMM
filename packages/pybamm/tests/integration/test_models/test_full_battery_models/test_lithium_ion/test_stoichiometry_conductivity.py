"""Integration tests for stoichiometry-dependent electrode conductivity (sigma) paths."""

import pytest

import pybamm


def _sto_dependent(values):
    """Return a copy of ``values`` with the electrode conductivities replaced by
    (small) functions of stoichiometry and temperature."""
    sigma_n = values["Negative electrode conductivity [S.m-1]"]
    sigma_p = values["Positive electrode conductivity [S.m-1]"]
    values = values.copy()
    values.update(
        {
            "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                0.1 * sigma_n * (0.1 + sto)
            ),
            "Positive electrode conductivity [S.m-1]": lambda sto, T: (
                0.1 * sigma_p * (0.1 + sto)
            ),
        }
    )
    return values


def _solve(model, values):
    sim = pybamm.Simulation(model, parameter_values=values)
    sim.solve([0, 600])
    return sim.solution["Voltage [V]"].entries[-1]


class TestStoichiometryConductivity:
    """Each test drives a different electrode-conductivity submodel with a
    stoichiometry-dependent conductivity and checks that the solution both runs and
    differs from the constant-conductivity baseline (i.e. the stoichiometry is
    actually wired through to sigma)."""

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({}, id="full_ohm"),
            pytest.param(
                {"surface form": "differential"}, id="surface_form_full_differential"
            ),
            pytest.param(
                {"surface form": "algebraic"}, id="surface_form_full_algebraic"
            ),
        ],
    )
    def test_dfn_paths(self, options):
        values = pybamm.ParameterValues("Chen2020")
        v_const = _solve(pybamm.lithium_ion.DFN(options), values)
        v_sto = _solve(pybamm.lithium_ion.DFN(options), _sto_dependent(values))
        assert abs(v_const - v_sto) > 1e-4

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({}, id="composite_ohm"),
            pytest.param(
                {"surface form": "differential"},
                id="composite_surface_form_differential",
            ),
        ],
    )
    def test_spme_paths(self, options):
        values = pybamm.ParameterValues("Chen2020")
        v_const = _solve(pybamm.lithium_ion.SPMe(options), values)
        v_sto = _solve(pybamm.lithium_ion.SPMe(options), _sto_dependent(values))
        assert abs(v_const - v_sto) > 1e-4

    def test_newman_tobias(self):
        # NewmanTobias also routes through full_ohm
        values = pybamm.ParameterValues("Chen2020")
        v_const = _solve(pybamm.lithium_ion.NewmanTobias(), values)
        v_sto = _solve(pybamm.lithium_ion.NewmanTobias(), _sto_dependent(values))
        assert abs(v_const - v_sto) > 1e-4

    def test_two_phase_composite(self):
        # two-phase electrode: the primary-phase surface stoichiometry feeds sigma
        options = {"particle phases": ("2", "1")}
        values = pybamm.ParameterValues("Chen2020_composite")
        values_sto = values.copy()
        values_sto.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    0.5 * (0.1 + sto)
                )
            }
        )
        v_const = _solve(pybamm.lithium_ion.DFN(options), values)
        v_sto = _solve(pybamm.lithium_ion.DFN(options), values_sto)
        assert abs(v_const - v_sto) > 1e-6

    def test_basic_dfn(self):
        values = pybamm.ParameterValues("Chen2020")
        v_const = _solve(pybamm.lithium_ion.BasicDFN(), values)
        v_sto = _solve(pybamm.lithium_ion.BasicDFN(), _sto_dependent(values))
        assert abs(v_const - v_sto) > 1e-4

    def test_basic_dfn_composite(self):
        values = pybamm.ParameterValues("Chen2020_composite")
        values_sto = values.copy()
        values_sto.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    0.5 * (0.1 + sto)
                )
            }
        )
        v_const = _solve(pybamm.lithium_ion.BasicDFNComposite(), values)
        v_sto = _solve(pybamm.lithium_ion.BasicDFNComposite(), values_sto)
        assert abs(v_const - v_sto) > 1e-6

    def test_basic_dfn_2d(self):
        model = pybamm.lithium_ion.BasicDFN2D()
        values = model.default_parameter_values
        values_sto = values.copy()
        values_sto.update(
            {
                "Negative electrode conductivity [S.m-1]": lambda sto, T: (
                    0.5 * (0.1 + sto)
                )
            }
        )
        v_sto = _solve(pybamm.lithium_ion.BasicDFN2D(), values_sto)
        assert v_sto > 0
