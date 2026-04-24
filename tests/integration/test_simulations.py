import casadi
import numpy as np
import pytest

import pybamm


class TestSimulationPickleRoundtrip:
    """Verify that pickling a Simulation and re-solving produces
    bit-exact identical raw solver output (sol.t, sol.y, sol.yp)."""

    @pytest.mark.parametrize("param_set", ["Chen2020", "Ai2020", "Marquis2019"])
    @pytest.mark.parametrize("initial_soc", [None, 0.5, "3.5 V"])
    @pytest.mark.parametrize(
        "use_experiment",
        [False, True],
    )
    def test_pickle_roundtrip_exact(
        self, tmp_path, param_set, initial_soc, use_experiment
    ):
        model = pybamm.lithium_ion.DFN()
        pv = pybamm.ParameterValues(param_set)
        if use_experiment:
            experiment = pybamm.Experiment(
                ["Discharge at C/2 until 3.2 V", "Hold at 3.2 V for 10 seconds"]
            )
            t_eval = None
        else:
            experiment = None
            t_eval = [0, 3600]
        sim = pybamm.Simulation(model, parameter_values=pv, experiment=experiment)

        sol_orig_full = sim.solve(t_eval=t_eval, initial_soc=initial_soc)

        filepath = str(tmp_path / "sim.pkl")
        sim.save(filepath)
        sim_loaded = pybamm.load_sim(filepath)

        sol_loaded_full = sim_loaded.solve(t_eval=t_eval, initial_soc=initial_soc)

        for i in range(len(sol_orig_full.all_ts)):
            sol_orig = sol_orig_full.sub_solutions[i]
            sol_loaded = sol_loaded_full.sub_solutions[i]
            tag = f"[{param_set}, segment={i}, soc={initial_soc}, use_experiment={use_experiment}]"

            np.testing.assert_array_equal(
                sol_orig.t, sol_loaded.t, err_msg=f"{tag} sol.t mismatch"
            )
            np.testing.assert_array_equal(
                sol_orig.y, sol_loaded.y, err_msg=f"{tag} sol.y mismatch"
            )
            if sol_orig.yp is not None:
                np.testing.assert_array_equal(
                    sol_orig.yp, sol_loaded.yp, err_msg=f"{tag} sol.yp mismatch"
                )


class TestSimulationConsistentState:
    def test_cv_initial_guess_uses_previous_current(self):
        charge_current = 15.0
        experiment = pybamm.Experiment(
            [
                (
                    f"Charge at {charge_current}A until 4.2 V",
                    "Hold at 4.2 V for 2 seconds",
                )
            ]
        )
        sim = pybamm.Simulation(
            pybamm.lithium_ion.DFN(),
            parameter_values=pybamm.ParameterValues("Chen2020"),
            experiment=experiment,
        )
        sol = sim.solve(initial_soc=0.05)

        # The mapper only runs when the two steps build distinct models.
        cc_step, cv_step = sim.experiment.steps
        cc_model = sim.steps_to_built_models[cc_step.basic_repr()]
        cv_model = sim.steps_to_built_models[cv_step.basic_repr()]
        assert cc_model is not cv_model

        cc_solution = sol.cycles[0].steps[0]

        # Replay BaseSolver.step's mapper invocation exactly: pull the
        # compiled (func, jac_y, jac_p) tuple, build the CV step's input
        # vector through the same code path, and call the func on the CC
        # step's last raw state vector.
        mapper_func, _, _ = sim._compiled_model_state_mappers[(cc_model, cv_model)]
        cv_inputs_dict = sim._build_experiment_step_inputs(
            {},
            cv_step,
            float(cc_solution.t[-1]),
            None,
            include_temperature=False,
        )
        cv_solver = sim._get_built_experiment_solver(cv_step)
        model_inputs = cv_solver._set_up_model_inputs(cv_model, cv_inputs_dict)
        p_vec = casadi.vertcat(*model_inputs.values())
        y_from = cc_solution.last_state.all_ys[0]
        seed = np.asarray(mapper_func(float(cc_solution.t[-1]), y_from, p_vec)).ravel()

        # Decode the CV model's "Current variable [A]" entry from the seed.
        current_var = next(
            v for v in cv_model.y_slices if v.name == "Current variable [A]"
        )
        current_slice = cv_model.y_slices[current_var][0]
        seed_current_scaled = float(seed[current_slice][0])
        seed_current_phys = float(current_var.reference.evaluate()) + (
            float(current_var.scale.evaluate()) * seed_current_scaled
        )

        assert seed_current_phys == pytest.approx(-charge_current, rel=1e-9)
