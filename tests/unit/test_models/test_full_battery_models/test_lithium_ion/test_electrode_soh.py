#
# Tests for the lithium-ion electrode-specific SOH model
#

import pytest

import pybamm


# Fixture for TestElectrodeSOHMSMR, TestCalculateTheoreticalEnergy and TestGetInitialOCPMSMR class.
@pytest.fixture()
def options():
    options = {
        "open-circuit potential": "MSMR",
        "particle": "MSMR",
        "number of MSMR reactions": ("6", "4"),
        "intercalation kinetics": "MSMR",
    }
    return options


class TestElectrodeSOH:
    def test_known_solution(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=None, param=param
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)

        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        assert sol["Up(y_100) - Un(x_100)"] == pytest.approx(Vmax, abs=1e-05)
        assert sol["Up(y_0) - Un(x_0)"] == pytest.approx(Vmin, abs=1e-05)
        assert sol["Q_Li"] == pytest.approx(Q_Li, abs=1e-05)

        # Solve with split esoh and check outputs
        ics = esoh_solver._set_up_solve(inputs, None)
        sol_split = esoh_solver._solve_split(inputs, ics, None)
        for key in sol:
            if key != "Maximum theoretical energy [W.h]":
                assert sol[key] == pytest.approx(sol_split[key].data[0], abs=1e-05)
            else:
                # theoretical_energy is not present in sol_split
                inputs = {
                    k: sol_split[k].data[0]
                    for k in ["x_0", "y_0", "x_100", "y_100", "Q_p"]
                }
                energy = esoh_solver.theoretical_energy_integral(inputs)
                assert sol[key] == pytest.approx(energy, abs=1e-05)

    def test_known_solution_cell_capacity(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=None, param=param, known_value="cell capacity"
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q = parameter_values.evaluate(param.Q)

        inputs = {"Q": Q, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        assert sol["Up(y_100) - Un(x_100)"] == pytest.approx(Vmax, abs=1e-05)
        assert sol["Up(y_0) - Un(x_0)"] == pytest.approx(Vmin, abs=1e-05)
        assert sol["Q"] == pytest.approx(Q, abs=1e-05)

    def test_error(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Ai2020")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=None, param=param
        )

        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q_Li = 2 * (Q_n + Q_p)

        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        with pytest.raises(ValueError, match="outside the range"):
            esoh_solver.solve(inputs)

        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)
        parameter_values.update(
            {
                "Open-circuit voltage at 0% SOC [V]": 0,
                "Open-circuit voltage at 100% SOC [V]": 5,
            }
            # need to update both the target voltages at 0 and 100% SOC
        )
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=None, param=param
        )
        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        # Solver fails to find a solution but voltage limits are not violated
        with pytest.raises(
            pybamm.SolverError, match="Could not find acceptable solution"
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
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=None, param=param
        )
        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        with pytest.raises(ValueError, match="upper bound of the voltage"):
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
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=None, param=param
        )
        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}
        with pytest.raises(ValueError, match="lower bound of the voltage"):
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
            parameter_values, direction=None, param=param, known_value="cell capacity"
        )
        with pytest.raises(ValueError, match="solve_for must be "):
            esoh_solver._get_electrode_soh_sims_split(None)

        inputs = {"Q_n": Q_n, "Q_p": Q_p, "Q": 2 * Q_p}
        with pytest.raises(
            ValueError, match="larger than the maximum possible capacity"
        ):
            esoh_solver.solve(inputs)


class TestElectrodeSOHComposite:
    @staticmethod
    def _check_phases_equal(results, xy, soc):
        assert results[f"{xy}_{soc}_1"] == pytest.approx(
            results[f"{xy}_{soc}_2"], abs=1e-05
        )

    @staticmethod
    def _get_params_and_options(composite_electrode):
        params = pybamm.ParameterValues("Chen2020")
        if composite_electrode == "negative" or composite_electrode == "both":
            phases = ("2", "1")
            params.update(
                {
                    "Primary: Negative electrode OCP [V]": params[
                        "Negative electrode OCP [V]"
                    ],
                    "Secondary: Negative electrode OCP [V]": params[
                        "Negative electrode OCP [V]"
                    ],
                    "Primary: Negative electrode active material volume fraction": 0.5,
                    "Secondary: Negative electrode active material volume fraction": 0.5,
                    "Primary: Maximum concentration in negative electrode [mol.m-3]": params[
                        "Maximum concentration in negative electrode [mol.m-3]"
                    ],
                    "Secondary: Maximum concentration in negative electrode [mol.m-3]": params[
                        "Maximum concentration in negative electrode [mol.m-3]"
                    ],
                    "Primary: Initial concentration in negative electrode [mol.m-3]": params[
                        "Initial concentration in negative electrode [mol.m-3]"
                    ],
                    "Secondary: Initial concentration in negative electrode [mol.m-3]": params[
                        "Initial concentration in negative electrode [mol.m-3]"
                    ],
                },
                check_already_exists=False,
            )
        if composite_electrode == "positive" or composite_electrode == "both":
            phases = ("1", "2")
            params.update(
                {
                    "Primary: Positive electrode OCP [V]": params[
                        "Positive electrode OCP [V]"
                    ],
                    "Secondary: Positive electrode OCP [V]": params[
                        "Positive electrode OCP [V]"
                    ],
                    "Primary: Positive electrode active material volume fraction": 0.5,
                    "Secondary: Positive electrode active material volume fraction": 0.5,
                    "Primary: Maximum concentration in positive electrode [mol.m-3]": params[
                        "Maximum concentration in positive electrode [mol.m-3]"
                    ],
                    "Secondary: Maximum concentration in positive electrode [mol.m-3]": params[
                        "Maximum concentration in positive electrode [mol.m-3]"
                    ],
                    "Primary: Initial concentration in positive electrode [mol.m-3]": params[
                        "Initial concentration in positive electrode [mol.m-3]"
                    ],
                    "Secondary: Initial concentration in positive electrode [mol.m-3]": params[
                        "Initial concentration in positive electrode [mol.m-3]"
                    ],
                },
                check_already_exists=False,
            )
        if composite_electrode == "both":
            phases = ("2", "2")
        options = {"particle phases": phases}
        return params, options

    @pytest.mark.parametrize("initial_value", ["4.0 V", 0.5])
    @pytest.mark.parametrize(
        "composite_electrode",
        [
            "both",  # both electrodes composite
            "negative",  # negative-only composite
            "positive",  # positive-only composite
        ],
    )
    def test_half_cell_with_same_ocp_curves(self, composite_electrode, initial_value):
        pvals, options = self._get_params_and_options(composite_electrode)
        # Use composite ESOH helper to compute initial stoichiometries at a voltage
        param = pybamm.LithiumIonParameters(options=options)
        results = pybamm.lithium_ion.get_initial_stoichiometries_composite(
            "4.0 V", pvals, direction=None, param=param, options=options
        )
        # Ensure keys exist and values are equal for both phases (this is not how the equation is set, but should be true)
        if composite_electrode == "positive" or composite_electrode == "both":
            assert pybamm.lithium_ion.check_if_composite(options, "positive")
            self._check_phases_equal(results, "y", "init")
            self._check_phases_equal(results, "y", "100")
            self._check_phases_equal(results, "y", "0")
        if composite_electrode == "negative" or composite_electrode == "both":
            assert pybamm.lithium_ion.check_if_composite(options, "negative")
            self._check_phases_equal(results, "x", "init")
            self._check_phases_equal(results, "x", "100")
            self._check_phases_equal(results, "x", "0")

        pvals_set = pybamm.lithium_ion.set_initial_state(
            initial_value, pvals, param=param, options=options
        )
        if initial_value == "4.0 V":
            assert pvals_set.evaluate(
                param.p.prim.U(results["y_init_1"], param.T_ref)
                - param.n.prim.U(results["x_init_1"], param.T_ref)
            ) == pytest.approx(4.0, abs=1e-05)

    def test_chen2020_composite_defaults(self):
        pvals = pybamm.ParameterValues("Chen2020_composite")
        options = {"particle phases": ("2", "1")}
        param = pybamm.LithiumIonParameters(options=options)
        results = pybamm.lithium_ion.get_initial_stoichiometries_composite(
            "4.0 V", pvals, param=param, options=options, tol=1e-1, direction=None
        )
        # Basic sanity: solution includes expected variables and bounded stoichiometries
        for key, val in results.items():
            if key.startswith(("x_", "y_")):
                assert 0 <= val <= 1
        pvals_set = pybamm.lithium_ion.set_initial_state(
            "4.0 V", pvals, param=param, options=options, tol=1e-1
        )
        assert pvals_set.evaluate(
            param.p.prim.U(results["y_init_1"], param.T_ref)
            - param.n.prim.U(results["x_init_1"], param.T_ref)
        ) == pytest.approx(4.0, abs=1e-05)

    def test_chen2020_composite_defaults_hysteresis(self):
        pvals = pybamm.ParameterValues("Chen2020_composite")
        options = {
            "particle phases": ("2", "1"),
            "open-circuit potential": (("single", "current sigmoid"), "single"),
        }
        param = pybamm.LithiumIonParameters(options=options)
        results_discharge = pybamm.lithium_ion.get_initial_stoichiometries_composite(
            "4.0 V",
            pvals,
            param=param,
            options=options,
            tol=1e-1,
            direction="discharge",
        )
        results_charge = pybamm.lithium_ion.get_initial_stoichiometries_composite(
            "4.0 V", pvals, param=param, options=options, tol=1e-1, direction="charge"
        )
        # Basic sanity: solution includes expected variables and bounded stoichiometries
        for key, val in results_discharge.items():
            if key.startswith(("x_", "y_")):
                assert 0 <= val <= 1
                assert results_discharge[key] != results_charge[key]


class TestElectrodeSOHMSMR:
    def test_known_solution(self, options):
        param = pybamm.LithiumIonParameters(options=options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        direction = None
        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values, direction=direction, param=param, options=options
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q_Li = parameter_values.evaluate(param.Q_Li_particles_init)

        inputs = {"Q_Li": Q_Li, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        assert sol["Up(y_100) - Un(x_100)"] == pytest.approx(Vmax, abs=1e-05)
        assert sol["Up(y_0) - Un(x_0)"] == pytest.approx(Vmin, abs=1e-05)
        assert sol["Q_Li"] == pytest.approx(Q_Li, abs=1e-05)

        # Solve with split esoh and check outputs
        ics = esoh_solver._set_up_solve(inputs, direction)
        sol_split = esoh_solver._solve_split(inputs, ics, direction)
        for key in sol:
            if key != "Maximum theoretical energy [W.h]":
                assert sol[key] == pytest.approx(sol_split[key].data[0], abs=1e-05)

        # Check feasibility checks can be performed successfully
        esoh_solver._check_esoh_feasible(inputs, direction=direction)

    def test_known_solution_cell_capacity(self, options):
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values,
            direction=None,
            param=param,
            known_value="cell capacity",
            options=options,
        )

        Vmin = 2.8
        Vmax = 4.2
        Q_n = parameter_values.evaluate(param.n.Q_init)
        Q_p = parameter_values.evaluate(param.p.Q_init)
        Q = parameter_values.evaluate(param.Q)

        inputs = {"Q": Q, "Q_n": Q_n, "Q_p": Q_p}

        # Solve the model and check outputs
        sol = esoh_solver.solve(inputs)

        assert sol["Up(y_100) - Un(x_100)"] == pytest.approx(Vmax, abs=1e-05)
        assert sol["Up(y_0) - Un(x_0)"] == pytest.approx(Vmin, abs=1e-05)
        assert sol["Q"] == pytest.approx(Q, abs=1e-05)

    def test_error(self, options):
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            parameter_values,
            direction=None,
            param=param,
            known_value="cell capacity",
            options=options,
        )
        with pytest.raises(ValueError, match="solve_for must be "):
            esoh_solver._get_electrode_soh_sims_split(None)


class TestElectrodeSOHHalfCell:
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
        assert sol["Uw(x_100)"].data[0] == pytest.approx(V_max, abs=1e-05)
        assert sol["Uw(x_0)"].data[0] == pytest.approx(V_min, abs=1e-05)


class TestCalculateTheoreticalEnergy:
    def test_efficiency(self, options):
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
        assert discharge_energy < theoretical_energy
        assert 0 < discharge_energy
        assert 0 < theoretical_energy


class TestGetInitialSOC:
    def test_initial_soc(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        T = parameter_values.evaluate(param.T_ref)

        x100, y100 = pybamm.lithium_ion.get_initial_stoichiometries(
            1, parameter_values, param=param, direction=None
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        assert V == pytest.approx(4.2)

        x0, y0 = pybamm.lithium_ion.get_initial_stoichiometries(
            0, parameter_values, param=param, direction=None
        )
        V = parameter_values.evaluate(param.p.prim.U(y0, T) - param.n.prim.U(x0, T))
        assert V == pytest.approx(2.8)

        x, y = pybamm.lithium_ion.get_initial_stoichiometries(
            0.4, parameter_values, param=param, direction=None
        )
        assert x == x0 + 0.4 * (x100 - x0)
        assert y == y0 - 0.4 * (y0 - y100)

        x, y = pybamm.lithium_ion.get_initial_stoichiometries(
            "4 V", parameter_values, param=param, direction=None
        )
        T = parameter_values.evaluate(param.T_ref)
        V = parameter_values.evaluate(param.p.prim.U(y, T) - param.n.prim.U(x, T))
        assert V == pytest.approx(4)

    def test_min_max_stoich(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        T = parameter_values.evaluate(param.T_ref)

        x0, x100, y100, y0 = pybamm.lithium_ion.get_min_max_stoichiometries(
            parameter_values, param=param, direction=None
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        assert V == pytest.approx(4.2)
        V = parameter_values.evaluate(param.p.prim.U(y0, T) - param.n.prim.U(x0, T))
        assert V == pytest.approx(2.8)

        x0, x100, y100, y0 = pybamm.lithium_ion.get_min_max_stoichiometries(
            parameter_values,
            param=param,
            known_value="cell capacity",
            direction=None,
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        assert V == pytest.approx(4.2)
        V = parameter_values.evaluate(param.p.prim.U(y0, T) - param.n.prim.U(x0, T))
        assert V == pytest.approx(2.8)

    def test_initial_soc_cell_capacity(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        T = parameter_values.evaluate(param.T_ref)

        x100, y100 = pybamm.lithium_ion.get_initial_stoichiometries(
            1,
            parameter_values,
            param=param,
            known_value="cell capacity",
            direction=None,
        )
        V = parameter_values.evaluate(param.p.prim.U(y100, T) - param.n.prim.U(x100, T))
        assert V == pytest.approx(4.2)

    def test_error(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values_half_cell = pybamm.lithium_ion.DFN(
            {"working electrode": "positive"}
        ).default_parameter_values

        with pytest.raises(ValueError, match="Initial SOC should be between 0 and 1"):
            pybamm.lithium_ion.get_initial_stoichiometries(2, parameter_values, None)

        with pytest.raises(ValueError, match="outside the voltage limits"):
            pybamm.lithium_ion.get_initial_stoichiometries(
                "1 V", parameter_values, direction=None
            )

        with pytest.raises(ValueError, match="must be a float"):
            pybamm.lithium_ion.get_initial_stoichiometries(
                "5 A", parameter_values, direction=None
            )

        with pytest.raises(ValueError, match="outside the voltage limits"):
            pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                "1 V", parameter_values_half_cell
            )

        with pytest.raises(ValueError, match="must be a float"):
            pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                "5 A", parameter_values_half_cell
            )

        with pytest.raises(ValueError, match="Initial SOC should be between 0 and 1"):
            pybamm.lithium_ion.get_initial_stoichiometry_half_cell(
                2, parameter_values_half_cell
            )

        with pytest.raises(
            ValueError,
            match="Known value must be cell capacity or cyclable lithium capacity",
        ):
            pybamm.lithium_ion.ElectrodeSOHSolver(
                parameter_values, direction=None, known_value="something else"
            )

        with pytest.raises(
            ValueError,
            match="Known value must be cell capacity or cyclable lithium capacity",
        ):
            param_MSMR = pybamm.lithium_ion.MSMR(
                {"number of MSMR reactions": "3"}
            ).param
            pybamm.models.full_battery_models.lithium_ion.electrode_soh._ElectrodeSOHMSMR(
                None, param=param_MSMR, known_value="something else"
            )

        with pytest.raises(
            ValueError,
            match="Known value must be cell capacity or cyclable lithium capacity",
        ):
            pybamm.models.full_battery_models.lithium_ion.electrode_soh._ElectrodeSOH(
                None, known_value="something else"
            )


class TestGetInitialOCP:
    def test_get_initial_ocp(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            1, parameter_values, param=param, direction=None
        )
        assert Up - Un == pytest.approx(4.2)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            0, parameter_values, param=param, direction=None
        )
        assert Up - Un == pytest.approx(2.8)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            "4 V", parameter_values, param=param, direction=None
        )
        assert Up - Un == pytest.approx(4)

    def test_min_max_ocp(self):
        param = pybamm.LithiumIonParameters()
        parameter_values = pybamm.ParameterValues("Mohtat2020")

        Un_0, Un_100, Up_100, Up_0 = pybamm.lithium_ion.get_min_max_ocps(
            parameter_values, param=param, direction=None
        )
        assert Up_100 - Un_100 == pytest.approx(4.2)
        assert Up_0 - Un_0 == pytest.approx(2.8)


class TestGetInitialOCPMSMR:
    def test_get_initial_ocp(self, options):
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            1, parameter_values, param=param, options=options
        )
        assert Up - Un == pytest.approx(4.2, abs=1e-05)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            0, parameter_values, param=param, options=options, direction=None
        )
        assert Up - Un == pytest.approx(2.8, abs=1e-05)
        Un, Up = pybamm.lithium_ion.get_initial_ocps(
            "4 V", parameter_values, param=param, options=options
        )
        assert Up - Un == pytest.approx(4)

    def test_min_max_ocp(self, options):
        param = pybamm.LithiumIonParameters(options)
        parameter_values = pybamm.ParameterValues("MSMR_Example")

        Un_0, Un_100, Up_100, Up_0 = pybamm.lithium_ion.get_min_max_ocps(
            parameter_values, param=param, direction=None, options=options
        )
        assert Up_100 - Un_100 == pytest.approx(4.2)
        assert Up_0 - Un_0 == pytest.approx(2.8)

        Un_0, Un_100, Up_100, Up_0 = pybamm.lithium_ion.get_min_max_ocps(
            parameter_values, param=param, known_value="cell capacity", options=options
        )
        assert Up_100 - Un_100 == pytest.approx(4.2)
        assert Up_0 - Un_0 == pytest.approx(2.8)
