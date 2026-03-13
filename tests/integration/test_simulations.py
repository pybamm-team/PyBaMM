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
