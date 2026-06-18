#
# Tests for DiffSLExport class
#

import csv
import importlib.util
import logging
import time
from pathlib import Path

import numpy as np
import pytest

import pybamm

has_pydiffsol = importlib.util.find_spec("pydiffsol") is not None
logging.getLogger("diffsl").setLevel(logging.WARNING)

OUTPUT_DIFFSL = False
OUTPUT_PLOTS = False
OUTPUT_TIMING = False


@pytest.fixture(scope="session")
def timing_results():
    results = []
    yield results
    if not OUTPUT_TIMING:
        return
    csv_path = Path("diffsl_timing_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test-name",
                "diffsl compilation time",
                "pybamm solve time",
                "diffsl solve time",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    logging.getLogger().info(f"Timing results written to {csv_path.resolve()}")


class TestDiffSLExport:
    @staticmethod
    def _simple_experiment():
        return pybamm.Experiment(
            [
                "Discharge at C/20 for 10 minutes",
                "Rest for 10 minutes",
                "Charge at C/20 for 10 minutes",
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
    def test_models(self, model, inputs, request, timing_results):
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
            ode_solver=ds.bdf,
        )
        compilation_time = time.perf_counter() - t0
        logger.info(f"DiffSL Ode compilation time: {compilation_time:.5f} seconds")
        ode.rtol = 1e-4
        ode.atol = 1e-6
        if isinstance(model, pybamm.lithium_ion.DFN):
            ode.ic_options.armijo_constant = 1e-1
            ode.options.max_nonlinear_solver_iterations = 100

        t_eval = [0, 3600]
        t_interp = np.linspace(t_eval[0], t_eval[1], 100)

        solver = pybamm.IDAKLUSolver(output_variables=[output_variable])
        _soln_pybamm = solver.solve(
            model_disc, t_eval, t_interp=t_interp, inputs=pv_inputs
        ).y

        t0 = time.perf_counter()
        voltage_pybamm = solver.solve(
            model_disc, t_eval, t_interp=t_interp, inputs=pv_inputs
        )[output_variable].data
        pybamm_solve_time = time.perf_counter() - t0
        logger.info(f"Pybamm solve time: {pybamm_solve_time:.5f} seconds")

        _diffsol_solution = ode.solve_dense(ds_inputs, t_interp)
        t0 = time.perf_counter()
        diffsol_solution = ode.solve_dense(ds_inputs, t_interp)
        voltage_diffsol = diffsol_solution.ys
        diffsl_solve_time = time.perf_counter() - t0
        logger.info(f"DiffSL solve time: {diffsl_solve_time:.5f} seconds")
        if voltage_diffsol.shape[0] == 1:
            voltage_diffsol = voltage_diffsol.flatten()

        if OUTPUT_TIMING:
            timing_results.append(
                {
                    "test-name": request.node.name,
                    "diffsl compilation time": f"{compilation_time:.5f}",
                    "pybamm solve time": f"{pybamm_solve_time:.5f}",
                    "diffsl solve time": f"{diffsl_solve_time:.5f}",
                }
            )

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
    def test_simulations(self, model, inputs, experiment, request, timing_results):
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
        solver = pybamm.IDAKLUSolver(output_variables=[output_variable])

        sim = pybamm.Simulation(
            model,
            parameter_values=pv,
            experiment=experiment,
            experiment_model_mode="unified",
            solver=solver,
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
        if OUTPUT_DIFFSL:
            code_path = Path(f"{request.node.name}.txt")
            with open(code_path, "w") as f:
                f.write(diffsl_code)

        t0 = time.perf_counter()
        ode = ds.Ode(
            diffsl_code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            ode_solver=ds.bdf,
        )
        compilation_time = time.perf_counter() - t0
        logger.info(f"DiffSL Ode compilation time: {compilation_time:.5f} seconds")
        ode.rtol = 1e-4
        ode.atol = 1e-6
        if isinstance(model, pybamm.lithium_ion.DFN):
            ode.ic_options.armijo_constant = 1e-1
            ode.options.max_nonlinear_solver_iterations = 100

        t_eval = [0, 3600]
        t_interp = np.linspace(t_eval[0], t_eval[1], 100)
        for _i in range(2):
            t0 = time.perf_counter()
            if experiment is None:
                solution = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs)
            else:
                solution = sim.solve(inputs=pv_inputs)
                t_interp = solution.t

        voltage_pybamm = solution[output_variable].data
        pybamm_solve_time = time.perf_counter() - t0
        logger.info(f"Pybamm solve time: {pybamm_solve_time:.5f} seconds")

        _diffsol_solution = ode.solve_dense(ds_inputs, t_interp)
        t0 = time.perf_counter()
        diffsol_solution = ode.solve_dense(ds_inputs, t_interp)
        voltage_diffsol = diffsol_solution.ys
        diffsl_solve_time = time.perf_counter() - t0
        logger.info(f"DiffSL solve time: {diffsl_solve_time:.5f} seconds")
        if voltage_diffsol.shape[0] == 1:
            voltage_diffsol = voltage_diffsol.flatten()

        if experiment is not None:
            voltage_pybamm = np.interp(diffsol_solution.ts, solution.t, voltage_pybamm)

        if OUTPUT_TIMING:
            timing_results.append(
                {
                    "test-name": request.node.name,
                    "diffsl compilation time": f"{compilation_time:.5f}",
                    "pybamm solve time": f"{pybamm_solve_time:.5f}",
                    "diffsl solve time": f"{diffsl_solve_time:.5f}",
                }
            )

        if OUTPUT_PLOTS:
            import matplotlib.pyplot as plt

            figure_path = Path(f"{request.node.name}.png")
            fig, ax = plt.subplots()
            ax.plot(diffsol_solution.ts, voltage_pybamm, label="PyBaMM")
            ax.plot(diffsol_solution.ts, voltage_diffsol, label="DiffSL")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Voltage [V]")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figure_path, dpi=150)
            plt.close(fig)

        # Note this tolerance is quite loose, for SPM and SPMe (not DFN) with experiment
        # there are often 1-2 points with higher error just after events,
        # unsure why the discrepancy occurs, maybe interpolation near events.
        # Error away from events is below solver tolerances.
        np.testing.assert_allclose(voltage_pybamm, voltage_diffsol, rtol=2e-3)
