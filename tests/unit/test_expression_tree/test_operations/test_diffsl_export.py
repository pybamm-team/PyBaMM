#
# Tests for DiffSLExport class
#

import importlib.util
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
        [
            [],
            ["Current function [A]"],
        ],
        ids=["no_inputs", "with_current"],
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
        output_variable = "X-averaged negative particle concentration [mol.m-3]"
        output_variable = "X-averaged positive particle concentration [mol.m-3]"
        output_variable = "Porosity times concentration [mol.m-3]"
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
        print(
            f"Pybamm simulation creation time: {time.perf_counter() - t0:.5f} seconds"
        )
        t0 = time.perf_counter()
        diffsl_code = pybamm.DiffSLExport(model).to_diffeq(
            inputs=inputs, outputs=[output_variable]
        )
        print(f"DiffSL export time: {time.perf_counter() - t0:.5f} seconds")

        t0 = time.perf_counter()
        ode = ds.Ode(
            diffsl_code,
            matrix_type=ds.faer_sparse,
            scalar_type=ds.f64,
            linear_solver=ds.lu,
            method=ds.tr_bdf2,
        )
        print(f"DiffSL Ode compilation time: {time.perf_counter() - t0:.5f} seconds")
        ode.rtol = 1e-4
        ode.atol = 1e-6
        if isinstance(model, pybamm.lithium_ion.DFN):
            ode.ic_options.armijo_constant = 1e-1

        t_eval = [0, 2600]
        t_interp = np.linspace(t_eval[0], t_eval[1], 10)

        soln_pybamm = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs).y

        with open("pybamm_code.txt", "w") as f:
            for key, value in sim.built_model.rhs.items():
                f.write(f"{key.name}: {value}\n")
        t0 = time.perf_counter()
        voltage_pybamm = sim.solve(t_eval, t_interp=t_interp, inputs=pv_inputs)[
            output_variable
        ].data
        print(f"Pybamm solve time: {time.perf_counter() - t0:.5f} seconds")

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
        print(f"DiffSL solve time: {time.perf_counter() - t0:.5f} seconds")
        if voltage_diffsol.shape[0] == 1:
            voltage_diffsol = voltage_diffsol.flatten()
        np.testing.assert_allclose(voltage_pybamm, voltage_diffsol, rtol=1e-5)

    def test_ode(self):
        model = pybamm.BaseModel()

        x = pybamm.Variable("x")
        y = pybamm.Variable("y")

        dxdt = 4 * x - 2 * y
        dydt = 3 * x - y

        model.rhs = {x: dxdt, y: dydt}
        model.initial_conditions = {x: pybamm.Scalar(1), y: pybamm.Scalar(2)}
        model.variables = {"x": x, "y": y, "z": x + 4 * y}

        disc = pybamm.Discretisation()
        model = disc.process_model(model)

        model = pybamm.DiffSLExport(model)
        correct_export = "in = []\nxinput_i { \n  1.\n}\n\nyinput_i { \n  2.\n}\nu_i {\n  x = xinput_i,\n  y = yinput_i,\n}\nF_i {\n  ((4.0 * x_i) - (2.0 * y_i)),\n  ((3.0 * x_i) - y_i),\n}\nout_i {\n  x_i,\n  y_i,\n  (x_i + (4.0 * y_i)),\n}"
        assert correct_export == model.to_diffeq(inputs=[], outputs=["x", "y", "z"])

    def test_heat_equation(self):
        model = pybamm.BaseModel()

        x = pybamm.SpatialVariable("x", domain="rod", coord_sys="cartesian")
        T = pybamm.Variable("Temperature", domain="rod")
        k = pybamm.Parameter("Thermal diffusivity")

        N = -k * pybamm.grad(T)
        dTdt = -pybamm.div(N)
        model.rhs = {T: dTdt}

        model.boundary_conditions = {
            T: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Dirichlet"),
            }
        }

        model.initial_conditions = {T: x}
        model.variables = {"Temperature": T, "Heat flux": N}
        geometry = {"rod": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(10)}}}
        param = pybamm.ParameterValues({"Thermal diffusivity": 1.0})
        param.process_model(model)
        param.process_geometry(geometry)

        submesh_types = {"rod": pybamm.Uniform1DSubMesh}
        var_pts = {x: 10}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        spatial_methods = {"rod": pybamm.FiniteVolume()}
        disc = pybamm.Discretisation(mesh, spatial_methods)

        disc.process_model(model)

        model = pybamm.DiffSLExport(model)

        # The order of the matrix entries is not deterministic, so we need to check both possible options
        correct_export_option_1 = "in = []\nconstant0_ij {\n  (0,0): 2.0,\n  (1,1): 1.0,\n  (1,0): -1.0,\n  (2,2): 1.0,\n  (2,1): -1.0,\n  (3,3): 1.0,\n  (3,2): -1.0,\n  (4,4): 1.0,\n  (4,3): -1.0,\n  (5,5): 1.0,\n  (5,4): -1.0,\n  (6,6): 1.0,\n  (6,5): -1.0,\n  (7,7): 1.0,\n  (7,6): -1.0,\n  (8,8): 1.0,\n  (8,7): -1.0,\n  (9,9): 1.0,\n  (9,8): -1.0,\n  (10,9): -2.0,\n}\nconstant1_ij {\n  (0,1): 1.0,\n  (0,0): -3.0,\n  (1,2): 1.0,\n  (1,0): 1.0,\n  (1,1): -2.0,\n  (2,3): 1.0,\n  (2,1): 1.0,\n  (2,2): -2.0,\n  (3,4): 1.0,\n  (3,2): 1.0,\n  (3,3): -2.0,\n  (4,5): 1.0,\n  (4,3): 1.0,\n  (4,4): -2.0,\n  (5,6): 1.0,\n  (5,4): 1.0,\n  (5,5): -2.0,\n  (6,7): 1.0,\n  (6,5): 1.0,\n  (6,6): -2.0,\n  (7,8): 1.0,\n  (7,6): 1.0,\n  (7,7): -2.0,\n  (8,9): 1.0,\n  (8,7): 1.0,\n  (8,8): -2.0,\n  (9,8): 1.0,\n  (9,9): -3.0,\n}\ntemperatureinput_i { \n  0.5,\n  1.5,\n  2.5,\n  3.5,\n  4.5,\n  5.5,\n  6.5,\n  7.5,\n  8.5,\n  9.5\n}\nu_i {\n  temperature = temperatureinput_i,\n}\nvarying0_i {\n  (constant0_ij * temperature_j),\n}\nF_i {\n  (constant1_ij * temperature_j),\n}\nout_i {\n  -(varying0_i),\n  temperature_i,\n}"
        correct_export_option_2 = "in = []\nconstant0_ij {\n  (0,1): 1.0,\n  (0,0): -3.0,\n  (1,2): 1.0,\n  (1,0): 1.0,\n  (1,1): -2.0,\n  (2,3): 1.0,\n  (2,1): 1.0,\n  (2,2): -2.0,\n  (3,4): 1.0,\n  (3,2): 1.0,\n  (3,3): -2.0,\n  (4,5): 1.0,\n  (4,3): 1.0,\n  (4,4): -2.0,\n  (5,6): 1.0,\n  (5,4): 1.0,\n  (5,5): -2.0,\n  (6,7): 1.0,\n  (6,5): 1.0,\n  (6,6): -2.0,\n  (7,8): 1.0,\n  (7,6): 1.0,\n  (7,7): -2.0,\n  (8,9): 1.0,\n  (8,7): 1.0,\n  (8,8): -2.0,\n  (9,8): 1.0,\n  (9,9): -3.0,\n}\nconstant1_ij {\n  (0,0): 2.0,\n  (1,1): 1.0,\n  (1,0): -1.0,\n  (2,2): 1.0,\n  (2,1): -1.0,\n  (3,3): 1.0,\n  (3,2): -1.0,\n  (4,4): 1.0,\n  (4,3): -1.0,\n  (5,5): 1.0,\n  (5,4): -1.0,\n  (6,6): 1.0,\n  (6,5): -1.0,\n  (7,7): 1.0,\n  (7,6): -1.0,\n  (8,8): 1.0,\n  (8,7): -1.0,\n  (9,9): 1.0,\n  (9,8): -1.0,\n  (10,9): -2.0,\n}\ntemperatureinput_i { \n  0.5,\n  1.5,\n  2.5,\n  3.5,\n  4.5,\n  5.5,\n  6.5,\n  7.5,\n  8.5,\n  9.5\n}\nu_i {\n  temperature = temperatureinput_i,\n}\nvarying0_i {\n  (constant1_ij * temperature_j),\n}\nF_i {\n  (constant0_ij * temperature_j),\n}\nout_i {\n  -(varying0_i),\n  temperature_i,\n}"
        model_output = model.to_diffeq(inputs=[], outputs=["Heat flux", "Temperature"])
        assert correct_export_option_1 == str(
            model_output
        ) or correct_export_option_2 == str(model_output)
