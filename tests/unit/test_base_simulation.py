import numpy as np
import pytest

import pybamm


class TestBaseSimulationClassHierarchy:
    """Tests for the BaseSimulation / Simulation class hierarchy."""

    def test_base_simulation_instantiation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        assert sim.operating_mode in (
            pybamm.BaseSimulation.MODE_WITHOUT_EXPERIMENT,
            pybamm.BaseSimulation.MODE_DRIVE_CYCLE,
        )
        assert sim.model is not None

    def test_inheritance(self):
        assert issubclass(pybamm.Simulation, pybamm.BaseSimulation)

    def test_isinstance_checks(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        assert isinstance(sim, pybamm.BaseSimulation)
        assert isinstance(sim, pybamm.Simulation)

        base_sim = pybamm.BaseSimulation(model)
        assert isinstance(base_sim, pybamm.BaseSimulation)
        assert not isinstance(base_sim, pybamm.Simulation)

    def test_base_simulation_no_experiment_attrs(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        assert not hasattr(sim, "experiment")
        assert not hasattr(sim, "_built_experiment_model")
        assert not hasattr(sim, "steps_to_built_models")
        assert not hasattr(sim, "model_state_mappers")
        assert not hasattr(sim, "_experiment_model_mode")

    def test_base_simulation_no_experiment_param(self):
        model = pybamm.lithium_ion.SPM()
        with pytest.raises(TypeError):
            pybamm.BaseSimulation(model, experiment="Discharge at 1C for 10 seconds")

    def test_base_simulation_no_experiment_model_mode_param(self):
        model = pybamm.lithium_ion.SPM()
        with pytest.raises(TypeError):
            pybamm.BaseSimulation(model, experiment_model_mode="unified")


class TestBaseSimulationSolveBuild:
    """Tests for core solve/build on BaseSimulation directly."""

    def test_base_solve(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sol = sim.solve([0, 600])
        assert sol is not None
        assert sim.solution is not None

    def test_base_solve_simple_model(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        sim = pybamm.BaseSimulation(model)
        sol = sim.solve([0, 1])
        np.testing.assert_allclose(sol.y[0], np.exp(-sol.t), rtol=1e-4, atol=1e-4)

    def test_base_solve_with_crate(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model, C_rate=2)
        assert sim.C_rate == 2
        sol = sim.solve([0, 1800])
        assert sol is not None

    def test_base_build(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.build()
        assert sim.built_model is not None
        assert sim.mesh is not None

    def test_base_step(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.step(dt=10)
        assert sim.solution is not None
        sim.step(dt=10)
        assert len(sim.solution.t) > 2

    def test_base_solve_drive_cycle(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values

        drive_cycle = np.column_stack([np.linspace(0, 100, 10), np.ones(10)])
        current_interpolant = pybamm.Interpolant(
            drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t
        )
        param["Current function [A]"] = current_interpolant
        sim = pybamm.BaseSimulation(model, parameter_values=param)
        assert sim.operating_mode == pybamm.BaseSimulation.MODE_DRIVE_CYCLE
        sol = sim.solve()
        assert sol is not None


class TestSimulationDispatch:
    """Tests that Simulation delegates correctly to BaseSimulation."""

    def test_simulation_non_experiment_delegates_to_base_solve(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 600])
        assert sol is not None

    def test_simulation_experiment_uses_own_solve(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model, experiment=pybamm.Experiment(["Discharge at 1C for 10 seconds"])
        )
        assert sim.operating_mode == pybamm.Simulation.MODE_WITH_EXPERIMENT
        sol = sim.solve()
        assert sol is not None

    def test_simulation_save_at_cycles_rejected_without_experiment(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        with pytest.raises(ValueError, match="save_at_cycles"):
            sim.solve([0, 600], save_at_cycles=2)

    def test_simulation_starting_solution_rejected_without_experiment(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 600])
        with pytest.raises(ValueError, match="starting_solution"):
            sim.solve([0, 600], starting_solution=sol)


class TestBaseSimulationPickle:
    """Tests for pickling BaseSimulation."""

    def test_base_simulation_pickle(self, tmp_path):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.solve([0, 100])

        filename = tmp_path / "base_sim.pkl"
        sim.save(str(filename))

        loaded = pybamm.load(str(filename))
        assert isinstance(loaded, pybamm.BaseSimulation)
        assert loaded.solution is not None

    def test_simulation_pickle_backward_compat(self, tmp_path):
        """Simulation unpickle restores experiment defaults for missing attrs."""
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        sim.solve([0, 100])

        filename = tmp_path / "sim.pkl"
        sim.save(str(filename))

        loaded = pybamm.load(str(filename))
        assert isinstance(loaded, pybamm.Simulation)
        assert isinstance(loaded, pybamm.BaseSimulation)
        assert loaded.solution is not None
        # Experiment defaults should be restored
        assert loaded.model_state_mappers == {}
        assert loaded._compiled_model_state_mappers == {}


class TestBaseSimulationProperties:
    """Tests that all properties are accessible on BaseSimulation."""

    def test_base_simulation_properties(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.build()

        assert sim.model is not None
        assert sim.model_with_set_params is not None
        assert sim.built_model is not None
        assert sim.geometry is not None
        assert sim.parameter_values is not None
        assert sim.submesh_types is not None
        assert sim.mesh is not None
        assert sim.var_pts is not None
        assert sim.spatial_methods is not None
        assert sim.solver is not None

    def test_base_simulation_save_model(self, tmp_path):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.build()
        filename = str(tmp_path / "model.json")
        sim.save_model(filename)

    def test_simulation_save_model_with_experiment_raises(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model, experiment=pybamm.Experiment(["Discharge at 1C for 10 seconds"])
        )
        sim.solve()
        with pytest.raises(NotImplementedError, match="experiment"):
            sim.save_model("test.json")


class TestPlottingAcceptsBaseSimulation:
    """Tests that plotting functions work with BaseSimulation instances."""

    def test_quickplot_accepts_base_simulation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.solve([0, 100])

        solutions = pybamm.QuickPlot.preprocess_solutions(sim)
        assert len(solutions) == 1
        assert isinstance(solutions[0], pybamm.Solution)

    def test_quickplot_check_input_validity_base_simulation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.solve([0, 100])

        # Should not raise
        result = pybamm.QuickPlot.check_input_validity(sim)
        assert len(result) == 1
