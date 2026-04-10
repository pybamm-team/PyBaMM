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
        ds_inputs = []
        for input_name in inputs:
            pv_inputs[input_name] = 0.9 * pv[input_name]
            ds_inputs.append(0.9 * pv[input_name])
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
        diffsl_code = pybamm.DiffSLExport(model_disc).to_diffeq(
            outputs=[output_variable]
        )
        logger.info(f"DiffSL export time: {time.perf_counter() - t0:.5f} seconds")

        t0 = time.perf_counter()
        ode = ds.Ode(
            diffsl_code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            method=ds.tr_bdf2,
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
            pybamm_jac = model_disc.jac_rhs_algebraic_action_eval(
                0, y, np.array(ds_inputs), v
            )
            pybamm_dydt = model_disc.rhs_algebraic_eval(0, y, np.array(ds_inputs))
            diffsol_dydt = ode.rhs(np.array(ds_inputs), 0, y.flatten())
            diffsol_jac = ode.rhs_jac_mul(np.array(ds_inputs), 0, y.flatten(), v)
            np.testing.assert_allclose(
                pybamm_dydt.full().flatten(), diffsol_dydt, rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                pybamm_jac.full().flatten(), diffsol_jac, rtol=1e-5, atol=1e-8
            )

        ds_inputs = np.array(ds_inputs)
        t0 = time.perf_counter()
        voltage_diffsol = ode.solve_dense(np.array(ds_inputs), t_interp)
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

        if experiment is not None:
            pytest.skip("pydiffsol does not yet support DiffSL model-index N")

        pv = model.default_parameter_values.copy()
        pv_inputs = {}
        ds_inputs = []
        for input_name in inputs:
            pv_inputs[input_name] = 0.9 * pv[input_name]
            ds_inputs.append(0.9 * pv[input_name])
            pv[input_name] = "[input]"
        output_variable = "Voltage [V]"

        sim = pybamm.Simulation(model, parameter_values=pv)

        t0 = time.perf_counter()
        diffsl_code = pybamm.DiffSLExport(sim).to_diffeq(outputs=[output_variable])
        logger = logging.getLogger()
        logger.info(f"DiffSL export time: {time.perf_counter() - t0:.5f} seconds")

        model_disc = sim.built_model
        assert model_disc is not None

        t0 = time.perf_counter()
        ode = ds.Ode(
            diffsl_code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            method=ds.tr_bdf2,
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

        solution = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs)
        soln_pybamm = solution.y

        t0 = time.perf_counter()
        voltage_pybamm = solution[output_variable].data
        logger.info(f"Pybamm solve time: {time.perf_counter() - t0:.5f} seconds")

        n = model_disc.y0.shape[0]
        v = np.ones(n)
        for i in range(soln_pybamm.shape[1]):
            y = soln_pybamm[:, i]
            pybamm_jac = model_disc.jac_rhs_algebraic_action_eval(
                0, y, np.array(ds_inputs), v
            )
            pybamm_dydt = model_disc.rhs_algebraic_eval(0, y, np.array(ds_inputs))
            diffsol_dydt = ode.rhs(np.array(ds_inputs), 0, y.flatten())
            diffsol_jac = ode.rhs_jac_mul(np.array(ds_inputs), 0, y.flatten(), v)
            np.testing.assert_allclose(
                pybamm_dydt.full().flatten(), diffsol_dydt, rtol=1e-5, atol=1e-8
            )
            np.testing.assert_allclose(
                pybamm_jac.full().flatten(), diffsol_jac, rtol=1e-5, atol=1e-8
            )

        ds_inputs = np.array(ds_inputs)
        t0 = time.perf_counter()
        voltage_diffsol = ode.solve_dense(np.array(ds_inputs), t_interp)
        logger.info(f"DiffSL solve time: {time.perf_counter() - t0:.5f} seconds")
        if voltage_diffsol.shape[0] == 1:
            voltage_diffsol = voltage_diffsol.flatten()
        np.testing.assert_allclose(voltage_pybamm, voltage_diffsol, rtol=3e-4)
