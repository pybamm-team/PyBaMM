#
# Tests for the lithium-ion electrode-specific SOH model
#

import pybamm
import unittest


class TestElectrodeSOH(unittest.TestCase):
    def test_known_solution(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)

        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        self.assertAlmostEqual(sol["Up(y_100) - Un(x_100)"], Vmax, places=5)
        self.assertAlmostEqual(sol["Up(y_0) - Un(x_0)"], Vmin, places=5)
        self.assertAlmostEqual(sol["Q_Li"], Q_Li, places=5)

        # Solve with split esoh and check outputs
        ics = esoh_solver._set_up_solve(inputs)
        sol_split = esoh_solver._solve_split(inputs, ics)
        for key in sol:
            if key != "Maximum theoretical energy [W.h]":
                self.assertAlmostEqual(sol[key], sol_split[key].data[0], places=5)
            else:
                # theoretical_energy is not present in sol_split
                inputs = {
                    k: sol_split[k].data[0]
                    for k in ["x_0", "y_0", "x_100", "y_100", "Q_p"]
                }
                energy = esoh_solver.theoretical_energy_integral(inputs)
                self.assertAlmostEqual(sol[key], energy, places=5)

    def test_known_solution_cell_capacity(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param, known_value="cell capacity"
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q = parameter_values.evaluate(param.Q)

        inputs = {"Q": Q, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        self.assertAlmostEqual(sol["Up(y_100) - Un(x_100)"], Vmax, places=5)
        self.assertAlmostEqual(sol["Up(y_0) - Un(x_0)"], Vmin, places=5)
        self.assertAlmostEqual(sol["Q"], Q, places=5)

    def test_error(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Ai2020")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)

        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q_Li = 2 * (Q_n + Q_p)

        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        with self.assertRaisesRegex(ValueError, "outside the range"):
            esoh_solver.solve(inputs)

        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)
        parameter_values.update(
            {
                "Open-circuit voltage at 0% SOC [V]": 0,
                "Open-circuit voltage at 100% SOC [V]": 5,
            }
            # need to update both the target voltages at 0 and 100% SOC
        )
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        # Solver fails to find a solution but voltage limits are not violated
        with self.assertRaisesRegex(
            pybamm.SolverError, "Could not find acceptable solution"
        ):
            esoh_solver.solve(inputs)
        # Solver fails to find a solution due to upper voltage limit
        parameter_values.update(
            {
                "Lower voltage cut-off [V]": 0,
                "Upper voltage cut-off [V]": 6,
                "Open-circuit voltage at 0% SOC [V]": 0,
                "Open-circuit voltage at 100% SOC [V]": 6,
            }
        )
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        with self.assertRaisesRegex(ValueError, "upper bound of the voltage"):
            esoh_solver.solve(inputs)
        # Solver fails to find a solution due to lower voltage limit
        parameter_values.update(
            {
                "Lower voltage cut-off [V]": -10,
                "Upper voltage cut-off [V]": 5,
                "Open-circuit voltage at 0% SOC [V]": -10,
                "Open-circuit voltage at 100% SOC [V]": 5,
            }
        )
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values, param)
        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        with self.assertRaisesRegex(ValueError, "lower bound of the voltage"):
            esoh_solver.solve(inputs)

        # errors for cell capacity based solver
        parameter_values.update(
            {
                "Lower voltage cut-off [V]": 3,
                "Upper voltage cut-off [V]": 4.2,
                "Open-circuit voltage at 0% SOC [V]": 3,
                "Open-circuit voltage at 100% SOC [V]": 4.2,
            }
        )
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param, known_value="cell capacity"
        )
        with self.assertRaisesRegex(ValueError, "solve_for must be "):
            esoh_solver._get_electrode_soh_sims_split()

        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q": 2 * Q_p}
        with self.assertRaisesRegex(
            ValueError, "larger than the maximum possible capacity"
        ):
            esoh_solver.solve(inputs)


class TestElectrodeSOHMSMR(unittest.TestCase):
    def test_known_solution(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        param = pybamm.LithiumIonParameters(options=options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param, options=options
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)

        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        self.assertAlmostEqual(sol["Up(y_100) - Un(x_100)"], Vmax, places=5)
        self.assertAlmostEqual(sol["Up(y_0) - Un(x_0)"], Vmin, places=5)
        self.assertAlmostEqual(sol["Q_Li"], Q_Li, places=5)

        # Solve with split esoh and check outputs
        ics = esoh_solver._set_up_solve(inputs)
        sol_split = esoh_solver._solve_split(inputs, ics)
        for key in sol:
            if key != "Maximum theoretical energy [W.h]":
                self.assertAlmostEqual(sol[key], sol_split[key].data[0], places=5)

        # Check feasibility checks can be performed successfully
        esoh_solver._check_esoh_feasible(inputs)

    def test_known_solution_cell_capacity(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param, known_value="cell capacity", options=options
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q = parameter_values.evaluate(param.Q)

        inputs = {"Q": Q, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        self.assertAlmostEqual(sol["Up(y_100) - Un(x_100)"], Vmax, places=5)
        self.assertAlmostEqual(sol["Up(y_0) - Un(x_0)"], Vmin, places=5)
        self.assertAlmostEqual(sol["Q"], Q, places=5)

    def test_error(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, param, known_value="cell capacity", options=options
        )
        with self.assertRaisesRegex(ValueError, "solve_for must be "):
            esoh_solver._get_electrode_soh_sims_split()


class TestElectrodeSOHHalfCell(unittest.TestCase):
    def test_known_solution(self):
        model = pybamm.lithium_ion.ElectrodeSOHHalfCell()
        param = pybamm.LithiumIonParameters({"working electrode": "positive"})
        parameter_values = pybamm.ParameterValues("Xu2019")
        Q_w = parameter_values.evaluate(param.p.Q_init)
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        V_min = 3.5
        V_max = 4.2
        # Solve the model and check outputs
        sol = sim.solve([0], inputs={"Q_w": Q_w})
        self.assertAlmostEqual(sol["Uw(x_100)"].data[0], V_max, places=5)
        self.assertAlmostEqual(sol["Uw(x_0)"].data[0], V_min, places=5)


class TestCalculateTheoreticalEnergy(unittest.TestCase):
    def test_efficiency(self):
        model = pybamm.lithium_ion.DFN(options={"calculate discharge energy": "true"})
        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve([0, 3600], initial_soc=1.0)
        discharge_energy = sol["Discharge energy [W.h]"].entries[-1]
        theoretical_energy = (
            pybamm.lithium_ion.electrode_soh.calculate_theoretical_energy(
                parameter_values
            )
        )
        # Real energy should be less than discharge energy,
        # and both should be greater than 0
        self.assertLess(discharge_energy, theoretical_energy)
        self.assertLess(0, discharge_energy)
        self.assertLess(0, theoretical_energy)


class TestGetInitialSOC(unittest.TestCase):
    def test_initial_soc(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        T = parameter_values.evaluate(param.T_ref)

        x100, y100 = pybamm.lithium_ion.get_initial_stoichiometries(
            1, parameter_values, param
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        self.assertAlmostEqual(V, 4.2)

        x0, y0 = pybamm.lithium_ion.get_initial_stoichiometries(
            0, parameter_values, param
        )
        V = parameter_values.evaluate(param.p.prim.U(y0, T) - param.n.prim.U(x0, T))
        self.assertAlmostEqual(V, 2.8)

        x, y = pybamm.lithium_ion.get_initial_stoichiometries(
            0.4, parameter_values, param
        )
        self.assertEqual(x, x0 + 0.4 * (x100 - x0))
        self.assertEqual(y, y0 - 0.4 * (y0 - y100))

        x, y = pybamm.lithium_ion.get_initial_stoichiometries(
            "4 V", parameter_values, param
        )
        T = parameter_values.evaluate(param.T_ref)
        V = parameter_values.evaluate(param.p.prim.U(y, T) - param.n.prim.U(x, T))
        self.assertAlmostEqual(V, 4)

    def test_min_max_stoich(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        T = parameter_values.evaluate(param.T_ref)

        x0, x100, y100, y0 = pybamm.lithium_ion.get_min_max_stoichiometries(
            parameter_values, param
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        self.assertAlmostEqual(V, 4.2)
        V = parameter_values.evaluate(param.p.prim.U(y0, T) - param.n.prim.U(x0, T))
        self.assertAlmostEqual(V, 2.8)

        x0, x100, y100, y0 = pybamm.lithium_ion.get_min_max_stoichiometries(
            parameter_values,
            param,
            known_value="cell capacity",
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        self.assertAlmostEqual(V, 4.2)
        V = parameter_values.evaluate(param.p.prim.U(y0, T) - param.n.prim.U(x0, T))
        self.assertAlmostEqual(V, 2.8)

    def test_initial_soc_cell_capacity(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        T = parameter_values.evaluate(param.T_ref)

        x100, y100 = pybamm.lithium_ion.get_initial_stoichiometries(
            1, parameter_values, param, known_value="cell capacity"
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        self.assertAlmostEqual(V, 4.2)

    def test_error(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values_half_cell = pybamm.lithium_ion.DFN(
            {"working electrode": "positive"}
        ).default_parameter_values

        with self.assertRaisesRegex(
            ValueError, "Initial SOC should be between 0 and 1"
        ):
            pybamm.lithium_ion.get_initial_stoichiometries(2, parameter_values)

        with self.assertRaisesRegex(ValueError, "outside the voltage limits"):
            pybamm.lithium_ion.get_initial_stoichiometries("1 V", parameter_values)

        with self.assertRaisesRegex(ValueError, "must be a float"):
            pybamm.lithium_ion.get_initial_stoichiometries("5 A", parameter_values)

        with self.assertRaisesRegex(ValueError, "outside the voltage limits"):
            pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                "1 V", parameter_values_half_cell
            )

        with self.assertRaisesRegex(ValueError, "must be a float"):
            pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                "5 A", parameter_values_half_cell
            )

        with self.assertRaisesRegex(
            ValueError, "Initial SOC should be between 0 and 1"
        ):
            pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                2, parameter_values_half_cell
            )

        with self.assertRaisesRegex(
            ValueError, "Known value must be cell capacity or cyclable lithium capacity"
        ):
            pybamm.lithium_ion.ElectrodeSOHSolver(
                parameter_values, known_value="something else"
            )

        with self.assertRaisesRegex(
            ValueError, "Known value must be cell capacity or cyclable lithium capacity"
        ):
            param_MSMR = pybamm.lithium_ion.MSMR(
                {"number of MSMR reactions": "3"}
            ).param
            pybamm.models.full_battery_models.lithium_ion.electrode_soh._ElectrodeSOHMSMR(
                param=param_MSMR, known_value="something else"
            )

        with self.assertRaisesRegex(
            ValueError, "Known value must be cell capacity or cyclable lithium capacity"
        ):
            pybamm.models.full_battery_models.lithium_ion.electrode_soh._ElectrodeSOH(
                known_value="something else"
            )


class TestGetInitialOCP(unittest.TestCase):
    def test_get_initial_ocp(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        Un, Up = pybamm.lithium_ion.get_initial_ocps(1, parameter_values, param)
        self.assertAlmostEqual(Up - Un, 4.2)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(0, parameter_values, param)
        self.assertAlmostEqual(Up - Un, 2.8)
        Un, Up = pybamm.lithium_ion.get_initial_ocps("4 V", parameter_values, param)
        self.assertAlmostEqual(Up - Un, 4)

    def test_min_max_ocp(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        Un_0, Un_100, Up_100, Up_0 = pybamm.lithium_ion.get_min_max_ocps(
            parameter_values, param
        )
        self.assertAlmostEqual(Up_100 - Un_100, 4.2)
        self.assertAlmostEqual(Up_0 - Un_0, 2.8)


class TestGetInitialOCPMSMR(unittest.TestCase):
    def test_get_initial_ocp(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            1, parameter_values, param, options=options
        )
        self.assertAlmostEqual(Up - Un, 4.2, places=5)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            0, parameter_values, param, options=options
        )
        self.assertAlmostEqual(Up - Un, 2.8, places=5)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            "4 V", parameter_values, param, options=options
        )
        self.assertAlmostEqual(Up - Un, 4)

    def test_min_max_ocp(self):
        options = {
            "open-circuit potential": "MSMR",
            "particle": "MSMR",
            "number of MSMR reactions": ("6", "4"),
            "intercalation kinetics": "MSMR",
        }
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        Un_0, Un_100, Up_100, Up_0 = pybamm.lithium_ion.get_min_max_ocps(
            parameter_values, param, options=options
        )
        self.assertAlmostEqual(Up_100 - Un_100, 4.2)
        self.assertAlmostEqual(Up_0 - Un_0, 2.8)

        Un_0, Un_100, Up_100, Up_0 = pybamm.lithium_ion.get_min_max_ocps(
            parameter_values, param, known_value="cell capacity", options=options
        )
        self.assertAlmostEqual(Up_100 - Un_100, 4.2)
        self.assertAlmostEqual(Up_0 - Un_0, 2.8)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
