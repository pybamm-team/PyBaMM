"""End-to-end coverage for Rust adjoint dense-row assembly."""

import numpy as np

import pybamm


class TestRustAdjointDenseRows:
    def test_spme_uses_one_adjoint_sweep_for_voltage_row(self):
        # voltage-as-a-state gives SPMe the dense voltage row this test targets;
        # the option's default reverted to "false" on main (#5670).
        model = pybamm.lithium_ion.SPMe(options={"voltage as a state": "true"})
        model.convert_to_format = "rust"
        simulation = pybamm.Simulation(model)
        solution = simulation.solve(np.linspace(0.0, 600.0, 20))

        stats = simulation.solver._setup["rust_model"].jacobian_stats()
        assert stats["n_dense_rows"] == 1
        assert stats["dense_row_tape_instructions"] > 0
        assert stats["dense_row_entries"] > stats["n_colors"]

        t = np.linspace(0.0, 600.0, 25)
        np.testing.assert_allclose(
            solution["Voltage [V]"](t),
            solution["Voltage expression [V]"](t),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_pouch_cv_hold_splits_its_outlier_row(self):
        # Many current-collector rows sit just over the dense-row threshold, so
        # only the far wider CV constraint row may drive the split.
        model = pybamm.lithium_ion.DFN(
            {"current collector": "potential pair", "dimensionality": 2}
        )
        model.convert_to_format = "rust"
        simulation = pybamm.Simulation(
            model,
            var_pts={"x_n": 4, "x_s": 4, "x_p": 4, "r_n": 4, "r_p": 4, "y": 8, "z": 8},
            experiment=pybamm.Experiment(["Hold at 4.1 V until C/20"]),
        )
        simulation.build_for_experiment()
        built = next(
            m
            for key, m in simulation.steps_to_built_models.items()
            if key.startswith("Voltage")
        )
        solver = pybamm.IDAKLUSolver()
        solver.set_up(built, inputs={}, t_eval=np.array([0.0, 1.0]))

        stats = solver._setup["rust_model"].jacobian_stats()
        assert stats["n_dense_rows"] == 1
        assert stats["dense_row_tape_instructions"] > 0
        # unsplit, the constraint row alone would force a colour per column
        assert stats["n_colors"] < stats["dense_row_entries"] // 10
        assert stats["jac_lane_width"] > 1
