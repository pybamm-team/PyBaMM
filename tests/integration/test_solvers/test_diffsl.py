#
# Tests for DiffSLExport class
#

import importlib.util
import logging
import time

import numpy as np
import pytest

import pybamm

has_pydiffsol = importlib.util.find_spec("pydiffsol") is not None


class TestDiffSLExport:
    @staticmethod
    def _simple_experiment():
        return pybamm.Experiment(
            [
                "Discharge at C/20 for 10 minutes",
                "Rest for 10 minutes",
            ]
        )

    @pytest.mark.skipif(not has_pydiffsol, reason="pydiffsol is not installed")
    @pytest.mark.parametrize(
        "model",
        [
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.SPMe(),
            pybamm.lithium_ion.DFN(),
        ],
        ids=["SPM", "SPMe", "DFN"],
    )
    @pytest.mark.parametrize(
        "inputs",
        [[], ["Lower voltage cut-off [V]"], ["Upper voltage cut-off [V]"]],
        ids=["no_inputs", "with_lower_cutoff", "with_upper_cutoff"],
    )
    def test_models(self, model, inputs):
        import pydiffsol as ds

        pv = model.default_parameter_values.copy()
        pv_inputs = {}
        for input_name in inputs:
            input_value = (
                1.1 * pv[input_name]
                if "Upper voltage" in input_name
                else 0.9 * pv[input_name]
            )
            pv_inputs[input_name] = input_value
            pv[input_name] = "[input]"
        output_variable = "Voltage [V]"

        t0 = time.perf_counter()
        geometry = model.default_geometry
        model_param = pv.process_model(model, inplace=False)
        pv.process_geometry(geometry)
        spatial_methods = model.default_spatial_methods
        var_pts = model.default_var_pts
        submesh_types = model.default_submesh_types
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, spatial_methods)
        model_disc = disc.process_model(model_param, inplace=True)
        logger = logging.getLogger()
        logger.info(
            f"Pybamm simulation creation time: {time.perf_counter() - t0:.5f} seconds"
        )

        t0 = time.perf_counter()
        exporter = pybamm.DiffSLExport(model_disc)
        diffsl_code = exporter.to_diffeq(outputs=[output_variable])
        ds_inputs = exporter.map_inputs(pv_inputs, outputs=[output_variable])
        logger.info(f"DiffSL export time: {time.perf_counter() - t0:.5f} seconds")

        t0 = time.perf_counter()
        ode = ds.Ode(
            diffsl_code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            ode_solver=ds.tr_bdf2,
        )
        logger.info(
            f"DiffSL Ode compilation time: {time.perf_counter() - t0:.5f} seconds"
        )
        ode.rtol = 1e-4
        ode.atol = 1e-6
        if isinstance(model, pybamm.lithium_ion.DFN):
            ode.ic_options.armijo_constant = 1e-1

        t_eval = [0, 3600]
        t_interp = np.linspace(t_eval[0], t_eval[1], 100)

        solver = pybamm.IDAKLUSolver()
        soln_pybamm = solver.solve(
            model_disc, t_eval, t_interp=t_interp, inputs=pv_inputs
        ).y

        t0 = time.perf_counter()
        voltage_pybamm = solver.solve(
            model_disc, t_eval, t_interp=t_interp, inputs=pv_inputs
        )[output_variable].data
        logger.info(f"Pybamm solve time: {time.perf_counter() - t0:.5f} seconds")

        n = model_disc.y0.shape[0]
        v = np.ones(n)
        for i in range(soln_pybamm.shape[1]):
            y = soln_pybamm[:, i]
            pybamm_jac = model_disc.jac_rhs_algebraic_action_eval(0, y, ds_inputs, v)
            pybamm_dydt = model_disc.rhs_algebraic_eval(0, y, ds_inputs)
            diffsol_dydt = ode.rhs(ds_inputs, 0, y.flatten())
            diffsol_jac = ode.rhs_jac_mul(ds_inputs, 0, y.flatten(), v)
            np.testing.assert_allclose(
                pybamm_dydt.full().flatten(), diffsol_dydt, rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                pybamm_jac.full().flatten(), diffsol_jac, rtol=1e-5, atol=1e-8
            )

        t0 = time.perf_counter()
        diffsol_solution = ode.solve_dense(ds_inputs, t_interp)
        voltage_diffsol = diffsol_solution.ys
        logger.info(f"DiffSL solve time: {time.perf_counter() - t0:.5f} seconds")
        if voltage_diffsol.shape[0] == 1:
            voltage_diffsol = voltage_diffsol.flatten()
        np.testing.assert_allclose(voltage_pybamm, voltage_diffsol, rtol=3e-4)

    @pytest.mark.skipif(not has_pydiffsol, reason="pydiffsol is not installed")
    @pytest.mark.parametrize(
        "model",
        [
            pybamm.lithium_ion.SPM(),
            pybamm.lithium_ion.SPMe(),
            pybamm.lithium_ion.DFN(),
        ],
        ids=["SPM", "SPMe", "DFN"],
    )
    @pytest.mark.parametrize(
        "inputs",
        [[], ["Lower voltage cut-off [V]"], ["Upper voltage cut-off [V]"]],
        ids=["no_inputs", "with_lower_cutoff", "with_upper_cutoff"],
    )
    @pytest.mark.parametrize(
        "experiment",
        [
            None,
            pytest.param("simple", id="with_experiment"),
        ],
    )
    def test_simulations(self, model, inputs, experiment):
        import pydiffsol as ds

        pv = model.default_parameter_values.copy()
        pv_inputs = {}
        for input_name in inputs:
            input_value = (
                1.1 * pv[input_name]
                if "Upper voltage" in input_name
                else 0.9 * pv[input_name]
            )
            pv_inputs[input_name] = input_value
            pv[input_name] = "[input]"
        output_variable = "Voltage [V]"
        experiment = self._simple_experiment() if experiment == "simple" else None

        sim = pybamm.Simulation(
            model,
            parameter_values=pv,
            experiment=experiment,
            experiment_model_mode="unified",
        )

        t0 = time.perf_counter()
        exporter = pybamm.DiffSLExport(sim)
        diffsl_code = exporter.to_diffeq(outputs=[output_variable])
        # For experiment simulations, ambient temperature is injected as an
        # InputParameter; supply its default value so map_inputs can resolve it.
        map_inputs_dict = dict(pv_inputs)
        if experiment is not None:
            map_inputs_dict["Ambient temperature [K]"] = pv["Ambient temperature [K]"]
        ds_inputs = exporter.map_inputs(map_inputs_dict, outputs=[output_variable])
        logger = logging.getLogger()
        logger.info(f"DiffSL export time: {time.perf_counter() - t0:.5f} seconds")

        model_disc = (
            sim.built_model if experiment is None else sim._built_experiment_model
        )
        assert model_disc is not None

        t0 = time.perf_counter()
        ode = ds.Ode(
            diffsl_code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            ode_solver=ds.tr_bdf2,
        )
        logger.info(
            f"DiffSL Ode compilation time: {time.perf_counter() - t0:.5f} seconds"
        )
        ode.rtol = 1e-4
        ode.atol = 1e-6
        if isinstance(model, pybamm.lithium_ion.DFN):
            ode.ic_options.armijo_constant = 1e-1

        t_eval = [0, 3600]
        t_interp = np.linspace(t_eval[0], t_eval[1], 100)
        if experiment is None:
            solution = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs)
        else:
            solution = sim.solve(inputs=pv_inputs)
            t_interp = solution.t
        soln_pybamm = solution.y

        t0 = time.perf_counter()
        voltage_pybamm = solution[output_variable].data
        logger.info(f"Pybamm solve time: {time.perf_counter() - t0:.5f} seconds")

        if experiment is None:
            n = model_disc.y0.shape[0]
            v = np.ones(n)
            for i in range(soln_pybamm.shape[1]):
                y = soln_pybamm[:, i]
                pybamm_jac = model_disc.jac_rhs_algebraic_action_eval(
                    0, y, ds_inputs, v
                )
                pybamm_dydt = model_disc.rhs_algebraic_eval(0, y, ds_inputs)
                diffsol_dydt = ode.rhs(ds_inputs, 0, y.flatten())
                diffsol_jac = ode.rhs_jac_mul(ds_inputs, 0, y.flatten(), v)
                np.testing.assert_allclose(
                    pybamm_dydt.full().flatten(), diffsol_dydt, rtol=1e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    pybamm_jac.full().flatten(), diffsol_jac, rtol=1e-5, atol=1e-8
                )

        t0 = time.perf_counter()
        diffsol_solution = ode.solve_dense(ds_inputs, t_interp)
        voltage_diffsol = diffsol_solution.ys
        logger.info(f"DiffSL solve time: {time.perf_counter() - t0:.5f} seconds")
        if voltage_diffsol.shape[0] == 1:
            voltage_diffsol = voltage_diffsol.flatten()
        if experiment is not None:
            voltage_pybamm = np.interp(diffsol_solution.ts, solution.t, voltage_pybamm)
            np.testing.assert_allclose(voltage_pybamm, voltage_diffsol, rtol=1e-3)
        else:
            np.testing.assert_allclose(voltage_pybamm, voltage_diffsol, rtol=3e-4)
