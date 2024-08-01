"""
Tests for the batch_study.py
"""

import os
import pybamm
import unittest
from tempfile import TemporaryDirectory


class TestBatchStudy(unittest.TestCase):
    def test_solve(self):
        spm = pybamm.lithium_ion.SPM()
        spm_uniform = pybamm.lithium_ion.SPM({"particle": "uniform profile"})
        casadi_safe = pybamm.CasadiSolver(mode="safe")
        casadi_fast = pybamm.CasadiSolver(mode="fast")
        exp1 = pybamm.Experiment(
            [("Discharge at C/5 for 10 minutes", "Rest for 1 hour")]
        )
        exp2 = pybamm.Experiment(
            [("Discharge at C/20 for 10 minutes", "Rest for 1 hour")]
        )

        bs_false_only_models = pybamm.BatchStudy(
            models={"SPM": spm, "SPM uniform": spm_uniform}
        )
        bs_true_only_models = pybamm.BatchStudy(
            models={"SPM": spm, "SPM uniform": spm_uniform}, permutations=True
        )
        bs_false = pybamm.BatchStudy(
            models={"SPM": spm, "SPM uniform": spm_uniform},
            solvers={"casadi safe": casadi_safe, "casadi fast": casadi_fast},
            experiments={"exp1": exp1, "exp2": exp2},
        )
        bs_true = pybamm.BatchStudy(
            models={"SPM": spm, "SPM uniform": spm_uniform},
            solvers={"casadi safe": casadi_safe, "casadi fast": casadi_fast},
            experiments={"exp2": exp2},
            permutations=True,
        )

        # Tests for exceptions
        for name in pybamm.BatchStudy.INPUT_LIST:
            with self.assertRaises(ValueError):
                pybamm.BatchStudy(
                    models={"SPM": spm, "SPM uniform": spm_uniform}, **{name: {None}}
                )

        # Tests for None when only models are given with permutations=False
        bs_false_only_models.solve(t_eval=[0, 3600])
        self.assertEqual(2, len(bs_false_only_models.sims))

        # Tests for None when only models are given with permutations=True
        bs_true_only_models.solve(t_eval=[0, 3600])
        self.assertEqual(2, len(bs_true_only_models.sims))

        # Tests for BatchStudy when permutations=False
        bs_false.solve()
        bs_false.plot(show_plot=False)
        self.assertEqual(2, len(bs_false.sims))
        for num in range(len(bs_false.sims)):
            output_model = bs_false.sims[num].model.name
            models_list = [model.name for model in bs_false.models.values()]
            self.assertIn(output_model, models_list)

            output_solver = bs_false.sims[num].solver.name
            solvers_list = [solver.name for solver in bs_false.solvers.values()]
            self.assertIn(output_solver, solvers_list)

            output_experiment = bs_false.sims[num].experiment.steps
            experiments_list = [
                experiment.steps for experiment in bs_false.experiments.values()
            ]
            self.assertIn(output_experiment, experiments_list)

        # Tests for BatchStudy when permutations=True
        bs_true.solve()
        bs_true.plot(show_plot=False)
        self.assertEqual(4, len(bs_true.sims))
        for num in range(len(bs_true.sims)):
            output_model = bs_true.sims[num].model.name
            models_list = [model.name for model in bs_true.models.values()]
            self.assertIn(output_model, models_list)

            output_solver = bs_true.sims[num].solver.name
            solvers_list = [solver.name for solver in bs_true.solvers.values()]
            self.assertIn(output_solver, solvers_list)

            output_experiment = bs_true.sims[num].experiment.steps
            experiments_list = [
                experiment.steps for experiment in bs_true.experiments.values()
            ]
            self.assertIn(output_experiment, experiments_list)

    def test_create_gif(self):
        with TemporaryDirectory() as dir_name:
            bs = pybamm.BatchStudy({"spm": pybamm.lithium_ion.SPM()})
            with self.assertRaisesRegex(
                ValueError, "The simulations have not been solved yet."
            ):
                pybamm.BatchStudy(
                    models={
                        "SPM": pybamm.lithium_ion.SPM(),
                        "SPM uniform": pybamm.lithium_ion.SPM(
                            {"particle": "uniform profile"}
                        ),
                    }
                ).create_gif()
            bs.solve([0, 10])

            # Create a temporary file name
            test_file = os.path.join(dir_name, "batch_study_test.gif")

            # create a GIF before calling the plot method
            bs.create_gif(number_of_images=3, duration=1, output_filename=test_file)

            # create a GIF after calling the plot method
            bs.plot(show_plot=False)
            bs.create_gif(number_of_images=3, duration=1, output_filename=test_file)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
