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
        assert eis_sim.solution is None

    def test_eis_not_simulation(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        eis_sim = pybamm.EISSimulation(model)
        assert not isinstance(eis_sim, pybamm.Simulation)


class TestEISSolution:
    """Tests for the EISSolution class."""

    def test_eis_solution_construction(self):
        freqs = np.logspace(-2, 2, 10)
        z = np.random.randn(10) + 1j * np.random.randn(10)
        sol = pybamm.EISSolution(freqs, z)

        np.testing.assert_array_equal(sol.frequencies, freqs)
        np.testing.assert_array_equal(sol.impedance, z)
        assert sol.impedance.dtype == complex

    def test_eis_solution_isinstance(self):
        sol = pybamm.EISSolution(np.array([1.0]), np.array([1 + 1j]))
        assert isinstance(sol, pybamm.SolutionBase)
        assert isinstance(sol, pybamm.EISSolution)
        assert not isinstance(sol, pybamm.Solution)

    def test_time_series_solution_isinstance(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.Solution(t, y, pybamm.BaseModel(), {})
        assert isinstance(sol, pybamm.SolutionBase)
        assert isinstance(sol, pybamm.Solution)
        assert not isinstance(sol, pybamm.EISSolution)

    def test_eis_solution_timing(self):
        sol = pybamm.EISSolution(np.array([1.0]), np.array([1 + 1j]))
        sol.set_up_time = 0.5
        sol.solve_time = 1.5
        assert sol.total_time == 2.0

    def test_eis_solution_get_data_dict(self):
        freqs = np.array([1.0, 10.0, 100.0])
        z = np.array([1 + 0.5j, 2 + 1j, 3 + 1.5j])
        sol = pybamm.EISSolution(freqs, z)
        data = sol.get_data_dict()

        np.testing.assert_array_equal(data["Frequency [Hz]"], freqs)
        np.testing.assert_array_equal(data["Z_re [Ohm]"], z.real)
        np.testing.assert_array_equal(data["Z_im [Ohm]"], z.imag)

    def test_eis_solution_save_data_csv(self, tmp_path):
        freqs = np.array([1.0, 10.0])
        z = np.array([1 + 0.5j, 2 + 1j])
        sol = pybamm.EISSolution(freqs, z)
        filepath = tmp_path / "eis_data.csv"
        sol.save_data(str(filepath), to_format="csv")
        assert filepath.exists()

    def test_eis_solution_save_data_json(self, tmp_path):
        import json

        freqs = np.array([1.0, 10.0])
        z = np.array([1 + 0.5j, 2 + 1j])
        sol = pybamm.EISSolution(freqs, z)
        filepath = tmp_path / "eis_data.json"
        sol.save_data(str(filepath), to_format="json")
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert "Frequency [Hz]" in data

    def test_eis_solution_save_pickle(self, tmp_path):
        freqs = np.array([1.0, 10.0])
        z = np.array([1 + 0.5j, 2 + 1j])
        sol = pybamm.EISSolution(freqs, z)
        filepath = tmp_path / "eis_sol.pkl"
        sol.save(str(filepath))
        assert filepath.exists()
        loaded = pybamm.load(str(filepath))
        np.testing.assert_array_equal(loaded.impedance, z)

    def test_eis_solution_save_data_invalid_format(self, tmp_path):
        sol = pybamm.EISSolution(np.array([1.0]), np.array([1 + 1j]))
        with pytest.raises(ValueError, match="Unrecognised format"):
            sol.save_data(str(tmp_path / "bad.xyz"), to_format="xyz")

    def test_eis_solution_getitem(self):
        freqs = np.array([1.0, 10.0, 100.0])
        z = np.array([1 + 0.5j, 2 + 1j, 3 + 1.5j])
        sol = pybamm.EISSolution(freqs, z)

        np.testing.assert_array_equal(sol["Frequency [Hz]"], freqs)
        np.testing.assert_array_equal(sol["Impedance [Ohm]"], z)
        np.testing.assert_array_equal(sol["Z_re [Ohm]"], z.real)
        np.testing.assert_array_equal(sol["Z_im [Ohm]"], z.imag)

    def test_eis_solution_data_property(self):
        freqs = np.array([1.0, 10.0])
        z = np.array([1 + 0.5j, 2 + 1j])
        sol = pybamm.EISSolution(freqs, z)
        data = sol.data
        assert "Frequency [Hz]" in data
        assert "Impedance [Ohm]" in data
        assert "Z_re [Ohm]" in data
        assert "Z_im [Ohm]" in data


class TestEISSimulationSolve:
    """Tests for EIS frequency-domain solving."""

    def test_solve_direct(self):
        model = pybamm.lithium_ion.SPM(
            options={"surface form": "differential"}, name="SPM"
        )
        eis_sim = pybamm.EISSimulation(model)
        frequencies = np.logspace(-2, 2, 10)
        result = eis_sim.solve(frequencies)

        assert isinstance(result, pybamm.EISSolution)
        assert isinstance(result, pybamm.SolutionBase)
        assert result.impedance.shape == (10,)
        assert result.frequencies.shape == (10,)
        assert np.iscomplexobj(result.impedance)
        assert eis_sim.solution is not None
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
        result = eis_sim.solve(frequencies, inputs={"C_dl": 0.1})
        assert result.impedance.shape == (10,)

    def test_solve_with_initial_soc(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        eis_sim = pybamm.EISSimulation(model, parameter_values=parameter_values)
        frequencies = np.logspace(-2, 2, 10)

        z_05 = eis_sim.solve(frequencies, initial_soc=0.5)
        z_09 = eis_sim.solve(frequencies, initial_soc=0.9)

        assert z_05.impedance.shape == (10,)
        assert not np.allclose(z_05.impedance, z_09.impedance)

    def test_initial_soc_via_build_matches_solve(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        frequencies = np.logspace(-2, 2, 10)

        eis_sim1 = pybamm.EISSimulation(model, parameter_values=parameter_values)
        eis_sim1.build(initial_soc=0.5)
        z1 = eis_sim1.solve(frequencies)

        eis_sim2 = pybamm.EISSimulation(model, parameter_values=parameter_values)
        z2 = eis_sim2.solve(frequencies, initial_soc=0.5)

        np.testing.assert_allclose(z1.impedance, z2.impedance)

    def test_initial_soc_voltage_string(self):
        model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        eis_sim = pybamm.EISSimulation(model, parameter_values=parameter_values)
        frequencies = np.logspace(-2, 2, 10)

        result = eis_sim.solve(frequencies, initial_soc="3.8 V")
        assert result.impedance.shape == (10,)


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

        bc_value = next(iter(model.boundary_conditions.values()))
        assert bc_value["left"][0].value == 3
        assert bc_value["right"][0].value == 42
