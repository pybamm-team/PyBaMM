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
        [[], ["Current function [A]"], ["Ambient temperature [K]"]],
        ids=["no_inputs", "with_current", "with_temperature"],
    )
    def test_models(self, model, inputs):
        import pydiffsol as ds

        pv = model.default_parameter_values
        pv_inputs = {}
        ds_inputs = []
        for input in inputs:
            pv_inputs[input] = pv[input]
            ds_inputs.append(pv[input])
            pv[input] = "[input]"
        output_variable = "Terminal voltage [V]"
        t0 = time.perf_counter()
        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(
            model=model,
            parameter_values=pv,
            output_variables=[output_variable],
            solver=solver,
        )
        sim.build()
        logger = logging.getLogger()
        logger.info(
            f"Pybamm simulation creation time: {time.perf_counter() - t0:.5f} seconds"
        )
        t0 = time.perf_counter()
        diffsl_code = pybamm.DiffSLExport(model).to_diffeq(
            inputs=inputs, outputs=[output_variable]
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

        soln_pybamm = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs).y

        t0 = time.perf_counter()
        voltage_pybamm = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs)[
            output_variable
        ].data
        logger.info(f"Pybamm solve time: {time.perf_counter() - t0:.5f} seconds")

        n = sim.built_model.y0.shape[0]
        v = np.ones(n)
        for i in range(soln_pybamm.shape[1]):
            y = soln_pybamm[:, i]
            pybamm_jac = sim.built_model.jac_rhs_algebraic_action_eval(
                0, y, np.array(ds_inputs), v
            )
            pybamm_dydt = sim.built_model.rhs_algebraic_eval(0, y, np.array(ds_inputs))
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
