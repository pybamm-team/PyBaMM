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

bs_false_only_models = pybamm.BatchStudy(models={"SPM": spm, "DFN": dfn})
bs_true_only_models = pybamm.BatchStudy(
    models={"SPM": spm, "DFN": dfn}, permutations=True
)
bs_false = pybamm.BatchStudy(
    models={"SPM": spm, "DFN": dfn},
    solvers={"casadi safe": casadi_safe, "casadi fast": casadi_fast},
    experiments={"cccv": cccv, "gitt": gitt},
)
bs_true = pybamm.BatchStudy(
    models={"SPM": spm, "DFN": dfn},
    solvers={"casadi safe": casadi_safe, "casadi fast": casadi_fast},
    experiments={"gitt": gitt},
    permutations=True,
)


class TestBatchStudy(unittest.TestCase):
    def test_solve(self):
        # Tests for exceptions
        for name in pybamm.BatchStudy.INPUT_LIST:
            with self.assertRaises(ValueError):
                pybamm.BatchStudy(
                    models={"SPM": spm, "DFN": dfn},
                    **{name: {None}}
                )

        # Tests for None when only models are given with permutations=False
        bs_false_only_models.solve(t_eval=[0, 3600])
        self.assertEqual(2, len(bs_false_only_models.sims))

        # Tests for None when only models are given with permutations=True
        bs_true_only_models.solve(t_eval=[0, 3600])
        self.assertEqual(2, len(bs_true_only_models.sims))

        # Tests for BatchStudy when permutations=False
        bs_false.solve()
        self.assertEqual(2, len(bs_false.sims))
        for num in range(len(bs_false.sims)):
            output_model = bs_false.sims[num].model.name
            models_list = [model.name for model in bs_false.models.values()]
            self.assertIn(output_model, models_list)

            output_solver = bs_false.sims[num].solver.name
            solvers_list = [
                solver.name for solver in bs_false.solvers.values()
            ]
            self.assertIn(output_solver, solvers_list)

            output_experiment = bs_false.sims[
                num
            ].experiment.operating_conditions_strings
            experiments_list = [
                experiment.operating_conditions_strings
                for experiment in bs_false.experiments.values()
            ]
            self.assertIn(output_experiment, experiments_list)

        # Tests for BatchStudy when permutations=True
        bs_true.solve()
        self.assertEqual(4, len(bs_true.sims))
        for num in range(len(bs_true.sims)):
            output_model = bs_true.sims[num].model.name
            models_list = [model.name for model in bs_true.models.values()]
            self.assertIn(output_model, models_list)

            output_solver = bs_true.sims[num].solver.name
            solvers_list = [solver.name for solver in bs_true.solvers.values()]
            self.assertIn(output_solver, solvers_list)

            output_experiment = bs_true.sims[
                num
            ].experiment.operating_conditions_strings
            experiments_list = [
                experiment.operating_conditions_strings
                for experiment in bs_true.experiments.values()
            ]
            self.assertIn(output_experiment, experiments_list)

    def test_plot(self):
        # Tests for BatchStudy when permutations=False
        bs_false.solve()
        bs_false.plot(testing=True)
        self.assertEqual(2, len(bs_false.sims))
        for num in range(len(bs_false.sims)):
            output_model = bs_false.sims[num].all_models[0].name
            models_list = [model.name for model in bs_false.models.values()]
            self.assertIn(output_model, models_list)

        # Tests for BatchStudy when permutations=True
        bs_true.solve()
        bs_true.plot(testing=True)
        self.assertEqual(4, len(bs_true.sims))
        for num in range(len(bs_true.sims)):
            output_model = bs_true.sims[num].all_models[0].name
            models_list = [model.name for model in bs_true.models.values()]
            self.assertIn(output_model, models_list)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
