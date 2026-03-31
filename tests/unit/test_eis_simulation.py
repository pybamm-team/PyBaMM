import numpy as np
import pytest

import pybamm
from pybamm.simulation.eis_utils import SymbolReplacer


class TestEISSimulationClassHierarchy:
    """Tests for EISSimulation class hierarchy and instantiation."""

    def test_eis_instantiation(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        eis_sim = pybamm.EISSimulation(model)
        assert isinstance(eis_sim, pybamm.BaseSimulation)
        assert isinstance(eis_sim, pybamm.EISSimulation)
        assert eis_sim.eis_solution is None

    def test_eis_not_simulation(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        eis_sim = pybamm.EISSimulation(model)
        assert not isinstance(eis_sim, pybamm.Simulation)


class TestEISSimulationSolve:
    """Tests for EIS frequency-domain solving."""

    def test_solve_direct(self):
        model = pybamm.lithium_ion.SPM(
            options={"surface form": "differential"}, name="SPM"
        )
        eis_sim = pybamm.EISSimulation(model)
        frequencies = np.logspace(-2, 2, 10)
        impedance = eis_sim.solve(frequencies)

        assert impedance.shape == (10,)
        assert np.iscomplex(impedance).all() or impedance.dtype == complex
        assert eis_sim.eis_solution is not None
        assert eis_sim.solve_time is not None

    def test_solve_with_inputs(self):
        model = pybamm.lithium_ion.DFN(
            options={"working electrode": "positive", "surface form": "differential"}
        )
        parameter_values = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameter_values.update(
            {
                "Positive electrode double-layer capacity [F.m-2]": pybamm.InputParameter(
                    "C_dl"
                ),
            },
        )
        eis_sim = pybamm.EISSimulation(model, parameter_values=parameter_values)
        frequencies = np.logspace(-2, 2, 10)
        impedance = eis_sim.solve(frequencies, inputs={"C_dl": 0.1})
        assert impedance.shape == (10,)

    def test_solve_with_initial_soc(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        eis_sim = pybamm.EISSimulation(model, parameter_values=parameter_values)
        frequencies = np.logspace(-2, 2, 10)

        z_05 = eis_sim.solve(frequencies, initial_soc=0.5)
        z_09 = eis_sim.solve(frequencies, initial_soc=0.9)

        assert z_05.shape == (10,)
        assert not np.allclose(z_05, z_09)

    def test_initial_soc_via_build_matches_solve(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        frequencies = np.logspace(-2, 2, 10)

        eis_sim1 = pybamm.EISSimulation(model, parameter_values=parameter_values)
        eis_sim1.build(initial_soc=0.5)
        z1 = eis_sim1.solve(frequencies)

        eis_sim2 = pybamm.EISSimulation(model, parameter_values=parameter_values)
        z2 = eis_sim2.solve(frequencies, initial_soc=0.5)

        np.testing.assert_allclose(z1, z2)

    def test_initial_soc_voltage_string(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        eis_sim = pybamm.EISSimulation(model, parameter_values=parameter_values)
        frequencies = np.logspace(-2, 2, 10)

        z = eis_sim.solve(frequencies, initial_soc="3.8 V")
        assert z.shape == (10,)


class TestNyquistPlot:
    """Tests for Nyquist plotting."""

    def test_nyquist_plot_returns_fig_and_axes(self):
        import matplotlib

        matplotlib.use("Agg")

        data = np.array([1 + 0.5j, 2 + 1j, 3 + 1.5j])
        fig, ax = pybamm.nyquist_plot(data)
        assert fig is not None
        assert ax is not None

    def test_nyquist_plot_with_existing_axes(self):
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        _, ax = plt.subplots()
        data = np.array([1 + 0.5j, 2 + 1j, 3 + 1.5j])
        fig, returned_ax = pybamm.nyquist_plot(data, ax=ax)
        assert fig is None
        assert returned_ax is ax
        plt.close("all")

    def test_eis_nyquist_plot(self):
        import matplotlib

        matplotlib.use("Agg")

        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        eis_sim = pybamm.EISSimulation(model)
        eis_sim.solve(np.logspace(-2, 2, 5))
        fig, ax = eis_sim.nyquist_plot()
        assert fig is not None
        assert ax is not None

    def test_nyquist_plot_before_solve_raises(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        eis_sim = pybamm.EISSimulation(model)
        with pytest.raises(ValueError, match="has not been solved"):
            eis_sim.nyquist_plot()


class TestSymbolReplacer:
    """Tests for the SymbolReplacer utility."""

    def test_symbol_replacements(self):
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        replacer = SymbolReplacer({a: b, c: d})

        for symbol_in, symbol_out in [
            (a, b),
            (a + a, b + b),
            (2 * pybamm.sin(a), 2 * pybamm.sin(b)),
            (3 * b, 3 * b),
            (a + c, b + d),
        ]:
            assert replacer.process_symbol(symbol_in) == symbol_out

    def test_concatenation_replacement(self):
        var1 = pybamm.Variable("var 1", domain="dom 1")
        var2 = pybamm.Variable("var 2", domain="dom 2")
        var3 = pybamm.Variable("var 3", domain="dom 1")
        conc = pybamm.concatenation(var1, var2)

        replacer = SymbolReplacer({var1: var3})
        result = replacer.process_symbol(conc)
        assert result == pybamm.concatenation(var3, var2)

    def test_process_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        var1 = pybamm.Variable("var1", domain="test")
        var2 = pybamm.Variable("var2", domain="test")
        model.rhs = {var1: a * pybamm.grad(var1)}
        model.algebraic = {var2: c * var2}
        model.initial_conditions = {var1: b, var2: d}
        model.boundary_conditions = {
            var1: {"left": (c, "Dirichlet"), "right": (d, "Neumann")}
        }
        model.variables = {
            "var1": var1,
            "var2": var2,
            "grad_var1": pybamm.grad(var1),
            "d_var1": d * var1,
        }

        replacer = SymbolReplacer(
            {
                pybamm.Parameter("a"): pybamm.Scalar(4),
                pybamm.Parameter("b"): pybamm.Scalar(2),
                pybamm.Parameter("c"): pybamm.Scalar(3),
                pybamm.Parameter("d"): pybamm.Scalar(42),
            }
        )
        replacer.process_model(model)

        var1 = model.variables["var1"]
        assert isinstance(model.rhs[var1], pybamm.Multiplication)
        assert model.rhs[var1].children[0].value == 4

        var2 = model.variables["var2"]
        assert isinstance(model.algebraic[var2], pybamm.Multiplication)
        assert model.algebraic[var2].children[0].value == 3

        assert isinstance(model.initial_conditions[var1], pybamm.Scalar)
        assert model.initial_conditions[var1].value == 2

        bc_value = list(model.boundary_conditions.values())[0]
        assert bc_value["left"][0].value == 3
        assert bc_value["right"][0].value == 42
