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


class TestBaseSimulationFingerprinting:
    """Tests for _pv_fingerprint, _normalize_inputs, _compute_esoh_fingerprint."""

    def test_pv_fingerprint_numeric(self):
        pv = {"a": 1.0, "b": 2, "c": 3.5}
        fp = pybamm.BaseSimulation._pv_fingerprint(pv)
        assert isinstance(fp, tuple)
        # sorted keys
        assert fp[0] == ("a", 1.0)
        assert fp[1] == ("b", 2)
        assert fp[2] == ("c", 3.5)

    def test_pv_fingerprint_non_numeric_uses_id(self):
        obj = object()
        pv = {"key": obj}
        fp = pybamm.BaseSimulation._pv_fingerprint(pv)
        assert fp == (("key", id(obj)),)

    def test_pv_fingerprint_deterministic(self):
        pv = {"z": 1.0, "a": 2.0}
        fp1 = pybamm.BaseSimulation._pv_fingerprint(pv)
        fp2 = pybamm.BaseSimulation._pv_fingerprint(pv)
        assert fp1 == fp2

    def test_normalize_inputs_float_int(self):
        inputs = {"x": 1, "y": 2.5}
        result = pybamm.BaseSimulation._normalize_inputs(inputs)
        assert result == (("x", 1.0), ("y", 2.5))

    def test_normalize_inputs_ndarray(self):
        arr = np.array([1.0, 2.0, 3.0])
        inputs = {"arr": arr}
        result = pybamm.BaseSimulation._normalize_inputs(inputs)
        assert result == (("arr", arr.tobytes()),)

    def test_normalize_inputs_non_numeric_uses_id(self):
        obj = object()
        inputs = {"obj": obj}
        result = pybamm.BaseSimulation._normalize_inputs(inputs)
        assert result == (("obj", id(obj)),)

    def test_compute_esoh_fingerprint_fallback_on_exception(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        # Provide inputs so the fallback path uses _normalize_inputs
        inputs = {"x": 1.0}
        fp = sim._compute_esoh_fingerprint(0.5, None, inputs)
        assert fp[0] == 0.5
        assert fp[1] is None
        # fp[2] is pv fingerprint, fp[3] is evals (either model-specific or fallback)
        assert isinstance(fp, tuple)
        assert len(fp) == 4

    def test_compute_esoh_fingerprint_no_inputs(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        fp = sim._compute_esoh_fingerprint(0.8, "discharge", None)
        assert fp[0] == 0.8
        assert fp[1] == "discharge"


class TestBaseSimulationSetInitialState:
    """Tests for set_initial_state caching and cache_esoh=False path."""

    def test_set_initial_state_cache_skips_on_same_fingerprint(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.set_initial_state(0.5)
        assert sim._built_initial_soc == 0.5
        fp1 = sim._esoh_fingerprint
        # Calling again with same SOC should not change fingerprint
        sim.set_initial_state(0.5)
        assert sim._esoh_fingerprint == fp1

    def test_set_initial_state_no_cache(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model, cache_esoh=False)
        sim.set_initial_state(0.5)
        assert sim._built_initial_soc == 0.5
        assert sim._needs_ic_rebuild is True

    def test_set_initial_state_no_cache_skips_on_same(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model, cache_esoh=False)
        sim.set_initial_state(0.5)
        fp1 = sim._esoh_fingerprint
        sim._needs_ic_rebuild = False
        # Same SOC again — should short circuit
        sim.set_initial_state(0.5)
        assert sim._esoh_fingerprint == fp1
        # _needs_ic_rebuild should still be False since we short-circuited
        assert sim._needs_ic_rebuild is False

    def test_set_initial_state_different_soc_triggers_rebuild(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.set_initial_state(0.5)
        sim._needs_ic_rebuild = False
        sim.set_initial_state(0.8)
        assert sim._needs_ic_rebuild is True
        assert sim._built_initial_soc == 0.8


class TestBaseSimulationPrepareSolve:
    """Tests for _prepare_solve."""

    def test_prepare_solve_defaults(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        t_eval, solver, calc_esoh, callbacks, inputs = sim._prepare_solve(
            [0, 100], None, None, None, None
        )
        assert t_eval == [0, 100]
        assert solver is sim._solver
        assert inputs == {}

    def test_prepare_solve_custom_solver(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        custom_solver = pybamm.CasadiSolver()
        t_eval, solver, calc_esoh, callbacks, inputs = sim._prepare_solve(
            [0, 100], custom_solver, None, None, None
        )
        assert solver is custom_solver

    def test_prepare_solve_calc_esoh_warning(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        sim = pybamm.BaseSimulation(model)
        with pytest.warns(UserWarning, match="not suitable for calculating eSOH"):
            sim._prepare_solve([0, 1], None, True, None, None)


class TestBaseSimulationGetBuiltModels:
    """Tests for _get_built_models."""

    def test_get_built_models_empty_before_build(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        assert sim._get_built_models() == []

    def test_get_built_models_after_build(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.build()
        models = sim._get_built_models()
        assert len(models) == 1
        assert models[0] is sim._built_model


class TestBaseSimulationRecomputeIC:
    """Tests for _recompute_initial_conditions via build with changed SOC."""

    def test_recompute_ic_triggered_by_soc_change(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.solve([0, 100], initial_soc=0.8)
        # Solving again with different SOC triggers IC recompute
        sim.solve([0, 100], initial_soc=0.5)
        assert sim._built_initial_soc == 0.5
        assert sim._needs_ic_rebuild is False


class TestBaseSimulationDeprecations:
    """Tests for deprecated methods."""

    def test_set_parameters_deprecation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            sim.set_parameters()
        assert sim._model_with_set_params is not None

    def test_set_initial_soc_deprecation(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            sim.set_initial_soc(0.5, None)
        assert sim._built_initial_soc == 0.5


class TestBaseSimulationPlotErrors:
    """Tests for plot/create_gif/plot_voltage_components unsolved errors."""

    def test_plot_raises_when_unsolved(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.raises(ValueError, match="not been solved"):
            sim.plot()

    def test_create_gif_raises_when_unsolved(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.raises(ValueError, match="not been solved"):
            sim.create_gif()

    def test_plot_voltage_components_raises_when_unsolved(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.raises(ValueError, match="not been solved"):
            sim.plot_voltage_components()


class TestBaseSimulationSaveModelErrors:
    """Tests for save_model edge cases."""

    def test_save_model_not_built_raises(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.raises(NotImplementedError, match="discretised"):
            sim.save_model("test.json")

    def test_save_python_format_raises(self):
        model = pybamm.lithium_ion.SPM()
        model.convert_to_format = "python"
        sim = pybamm.BaseSimulation(model)
        sim.build()
        with pytest.raises(NotImplementedError, match="python"):
            sim.save("test.pkl")


class TestBaseSimulationSolveErrors:
    """Tests for solve error paths."""

    def test_solve_without_t_eval_raises(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        with pytest.raises(pybamm.SolverError, match="t_eval"):
            sim.solve()


class TestBaseSimulationBuildIdempotent:
    """Test that build is idempotent and handles pre-discretised models."""

    def test_build_already_discretised_model(self):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        sim = pybamm.BaseSimulation(model)
        sim.build()
        assert sim.built_model is not None

    def test_build_called_twice_is_noop(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.BaseSimulation(model)
        sim.build()
        built1 = sim.built_model
        sim.build()
        assert sim.built_model is built1
