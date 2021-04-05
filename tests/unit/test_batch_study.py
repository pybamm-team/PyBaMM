"""
Tests for the batch_study.py
"""
import pybamm
import unittest

spm = pybamm.lithium_ion.SPM()
dfn = pybamm.lithium_ion.DFN()
casadi_safe = pybamm.CasadiSolver(mode="safe")
casadi_fast = pybamm.CasadiSolver(mode="fast")
cccv = pybamm.Experiment(
    [
        (
            "Discharge at C/5 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 10 mA",
            "Rest for 1 hour",
        ),
    ]
    * 3
)
gitt = pybamm.Experiment(
    [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20,
)

batch_study_false_only_models = pybamm.BatchStudy(models={"SPM": spm, "DFN": dfn})
batch_study_true_only_models = pybamm.BatchStudy(
    models={"SPM": spm, "DFN": dfn}, permutations=True
)
batch_study_false = pybamm.BatchStudy(
    models={"SPM": spm, "DFN": dfn},
    solvers={"casadi safe": casadi_safe, "casadi fast": casadi_fast},
    experiments={"cccv": cccv, "gitt": gitt},
)
batch_study_true = pybamm.BatchStudy(
    models={"SPM": spm, "DFN": dfn},
    solvers={"casadi safe": casadi_safe, "casadi fast": casadi_fast},
    experiments={"gitt": gitt},
    permutations=True,
)


class TestBatchStudy(unittest.TestCase):
    def test_solve(self):
        # Tests for exceptions
        with self.assertRaises(ValueError):
            pybamm.BatchStudy(
                models={"SPM": spm, "DFN": dfn}, experiments={"gitt": gitt}
            )

        with self.assertRaises(ValueError):
            pybamm.BatchStudy(
                models={"SPM": spm, "DFN": dfn},
                solvers={"casadi fast": casadi_fast},
                experiments={"cccv": cccv, "gitt": gitt},
            )

        # Tests for batch_study_false_only_models (Only models with permutations=False)
        batch_study_false_only_models.solve(t_eval=[0, 3600])
        output_len_false_only_models = len(batch_study_false_only_models.sims)
        self.assertEqual(2, output_len_false_only_models)

        # Tests for batch_study_true_only_models (Only models with permutations=True)
        batch_study_true_only_models.solve(t_eval=[0, 3600])
        output_len_true_only_models = len(batch_study_true_only_models.sims)
        self.assertEqual(2, output_len_true_only_models)

        # Tests for batch_study_false (permutations=False)
        batch_study_false.solve()
        output_len_false = len(batch_study_false.sims)
        self.assertEqual(2, output_len_false)
        for num in range(output_len_false):
            output_model = batch_study_false.sims[num].model.name
            models_list = [model.name for model in batch_study_false.models.values()]
            self.assertIn(output_model, models_list)

            output_solver = batch_study_false.sims[num].solver.name
            solvers_list = [
                solver.name for solver in batch_study_false.solvers.values()
            ]
            self.assertIn(output_solver, solvers_list)

            output_experiment = batch_study_false.sims[
                num
            ].experiment.operating_conditions_strings
            experiments_list = [
                experiment.operating_conditions_strings
                for experiment in batch_study_false.experiments.values()
            ]
            self.assertIn(output_experiment, experiments_list)

        # Tests for batch_study_true (permutations=True)
        batch_study_true.solve()
        output_len_true = len(batch_study_true.sims)
        self.assertEqual(4, output_len_true)
        for num in range(output_len_true):
            output_model = batch_study_true.sims[num].model.name
            models_list = [model.name for model in batch_study_true.models.values()]
            self.assertIn(output_model, models_list)

            output_solver = batch_study_true.sims[num].solver.name
            solvers_list = [solver.name for solver in batch_study_true.solvers.values()]
            self.assertIn(output_solver, solvers_list)

            output_experiment = batch_study_true.sims[
                num
            ].experiment.operating_conditions_strings
            experiments_list = [
                experiment.operating_conditions_strings
                for experiment in batch_study_true.experiments.values()
            ]
            self.assertIn(output_experiment, experiments_list)

    def test_plot(self):
        # Tests for batch_study_false (permutations=False)
        batch_study_false.solve()
        batch_study_false.plot()
        output_len_false = len(batch_study_false.sims)
        self.assertEqual(2, output_len_false)
        for num in range(output_len_false):
            output_model = batch_study_false.sims[num].all_models[0].name
            models_list = [model.name for model in batch_study_false.models.values()]
            self.assertIn(output_model, models_list)

        # Tests for batch_study_true (permutations=True)
        batch_study_true.solve()
        batch_study_true.plot()
        output_len_true = len(batch_study_true.sims)
        self.assertEqual(4, output_len_true)
        for num in range(output_len_true):
            output_model = batch_study_true.sims[num].all_models[0].name
            models_list = [model.name for model in batch_study_true.models.values()]
            self.assertIn(output_model, models_list)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
