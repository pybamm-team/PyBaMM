import os

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import trapezoid

import pybamm
from pybamm.solvers.base_solver import BaseSolver
from tests import no_internet_connection


class TestSimulation:
    def test_simple_model(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 1])
        np.testing.assert_allclose(sol.y[0], np.exp(-sol.t), rtol=1e-4, atol=1e-4)

    def test_solve(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        sim.solve([0, 600])
        assert sim._solution is not None
        for val in list(sim.built_model.rhs.values()):
            assert not val.has_symbol_of_classes(pybamm.Parameter)
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                assert val.has_symbol_of_classes(pybamm.Matrix)

        # test solve without check
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(), discretisation_kwargs={"check_model": False}
        )
        sol = sim.solve(t_eval=[0, 600])
        for val in list(sim.built_model.rhs.values()):
            assert not val.has_symbol_of_classes(pybamm.Parameter)
            # skip test for scalar variables (e.g. discharge capacity)
            if val.size > 1:
                assert val.has_symbol_of_classes(pybamm.Matrix)

        # Test options that are only available when simulating an experiment
        with pytest.raises(ValueError, match=r"save_at_cycles"):
            sim.solve(save_at_cycles=2)
        with pytest.raises(ValueError, match=r"starting_solution"):
            sim.solve(starting_solution=sol)

    def test_solve_remove_independent_variables_from_rhs(self):
        sim = pybamm.Simulation(
            pybamm.lithium_ion.SPM(),
            discretisation_kwargs={"remove_independent_variables_from_rhs": True},
        )
        sol = sim.solve([0, 600])
        t = sol["Time [s]"].data
        I = sol["Current [A]"].data
        q = sol["Discharge capacity [A.h]"].data
        np.testing.assert_allclose(q, trapezoid(I, t) / 3600, rtol=1e-7, atol=1e-6)

    def test_solve_non_battery_model(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        model.variables = {"v": v}
        sim = pybamm.Simulation(
            model, solver=pybamm.ScipySolver(rtol=1e-10, atol=1e-10)
        )

        sim.solve(np.linspace(0, 1, 100))
        np.testing.assert_array_equal(sim.solution.t, np.linspace(0, 1, 100))
        np.testing.assert_allclose(
            sim.solution["v"].entries,
            np.exp(-np.linspace(0, 1, 100)),
            rtol=1e-7,
            atol=1e-6,
        )

    def test_solve_already_partially_processed_model(self):
        model = pybamm.lithium_ion.SPM()

        # Process model manually
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        # Let simulation take over
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])

        # Discretised manually
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        # Let simulation take over
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])

        # The model is still observable because it has not yet been processed by
        # the parameter_values or discretisation
        assert sim.solution.observable is True
        assert all(model.solution_observable for model in sim.solution.all_models)

    def test_reuse_commands(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        sim.set_parameters()
        sim.set_parameters()

        sim.build()
        sim.build()

        sim.solve([0, 600])
        sim.solve([0, 600])

        sim.build()
        sim.solve([0, 600])
        sim.set_parameters()

    def test_set_crate(self):
        model = pybamm.lithium_ion.SPM()
        current_1C = model.default_parameter_values["Current function [A]"]
        sim = pybamm.Simulation(model, C_rate=2)
        assert sim.parameter_values["Current function [A]"] == 2 * current_1C
        assert sim.C_rate == 2

    def test_step(self):
        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        sim.step(dt)  # 1 step stores first 12 points
        assert sim.solution.y[0, :].size == 12
        np.testing.assert_allclose(
            [sim.solution.t[0], sim.solution.t[-1]],
            np.array([0, dt]),
            rtol=1e-7,
            atol=1e-6,
        )
        saved_sol = sim.solution

        sim.step(dt)  # automatically append the next step
        assert sim.solution.y[0, :].size == 24
        np.testing.assert_allclose(
            [sim.solution.t[0], sim.solution.t[-1]],
            np.array([0, 2 * dt]),
            rtol=1e-7,
            atol=1e-6,
        )

        sim.step(dt, save=False)  # now only store the two end step points
        assert sim.solution.y[0, :].size == 12
        np.testing.assert_allclose(
            [sim.solution.t[0], sim.solution.t[-1]],
            np.array([2 * dt + 1e-9, 3 * dt]),
            rtol=1e-7,
            atol=1e-6,
        )
        # Start from saved solution
        sim.step(dt, starting_solution=saved_sol)
        assert sim.solution.y[0, :].size == 24
        np.testing.assert_allclose(
            [sim.solution.t[0], sim.solution.t[-1]],
            np.array([0, 2 * dt]),
            rtol=1e-7,
            atol=1e-6,
        )

    @pytest.mark.skipif(
        no_internet_connection(),
        reason="Network not available to download files from registry",
    )
    def test_solve_with_initial_soc(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(t_eval=[0, 600], initial_soc=1)
        assert sim._built_initial_soc == 1
        sim.solve(t_eval=[0, 600], initial_soc=0.5)
        assert sim._built_initial_soc == 0.5
        exp = pybamm.Experiment(
            [pybamm.step.string("Discharge at 1C until 3.6V", period="1 minute")]
        )
        sim = pybamm.Simulation(model, parameter_values=param, experiment=exp)
        sim.solve(initial_soc=0.8)
        assert sim._built_initial_soc == 0.8

        # test with drive cycle
        data_loader = pybamm.DataLoader()
        drive_cycle = pd.read_csv(
            data_loader.get_data("US06.csv"),
            comment="#",
            header=None,
        ).to_numpy()
        current_interpolant = pybamm.Interpolant(
            drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t
        )
        param["Current function [A]"] = current_interpolant
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(initial_soc=0.8)
        assert sim._built_initial_soc == 0.8

        # Test that build works with initial_soc
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.build(initial_soc=0.5)
        assert sim._built_initial_soc == 0.5

        # Test that initial soc works with a relevant input parameter
        model = pybamm.lithium_ion.DFN()
        param = model.default_parameter_values
        og_eps_p = param["Positive electrode active material volume fraction"]
        param["Positive electrode active material volume fraction"] = (
            pybamm.InputParameter("eps_p")
        )
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(t_eval=[0, 1], initial_soc=0.8, inputs={"eps_p": og_eps_p})
        assert sim._built_initial_soc == 0.8

        # test having an input parameter in the ocv function
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        a = pybamm.Parameter("a")

        def ocv_with_parameter(sto):
            u_eq = (4.2 - 2.5) * (1 - sto) + 2.5
            return a * u_eq

        parameter_values.update(
            {
                "Positive electrode OCP [V]": ocv_with_parameter,
            }
        )
        parameter_values.update({"a": "[input]"}, check_already_exists=False)
        experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sim.solve([0, 3600], inputs={"a": 1})

        # Test whether initial_soc works with half cell (solve)
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.DFN(options)
        sim = pybamm.Simulation(model)
        sim.solve([0, 1], initial_soc=0.9)
        assert sim._built_initial_soc == 0.9

        # Test whether initial_soc works with half cell (build)
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.DFN(options)
        sim = pybamm.Simulation(model)
        sim.build(initial_soc=0.9)
        assert sim._built_initial_soc == 0.9

        # Test whether initial_soc works with half cell when it is a voltage
        model = pybamm.lithium_ion.SPM({"working electrode": "positive"})
        parameter_values = model.default_parameter_values
        ucv = parameter_values["Open-circuit voltage at 100% SOC [V]"]
        parameter_values["Open-circuit voltage at 100% SOC [V]"] = ucv + 1e-12
        parameter_values["Upper voltage cut-off [V]"] = ucv + 1e-12
        options = {"working electrode": "positive"}
        parameter_values["Current function [A]"] = 0.0
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve([0, 1], initial_soc="4.1 V")
        voltage = sol["Terminal voltage [V]"].entries
        assert voltage[0] == pytest.approx(4.1, abs=1e-05)

        # test with MSMR
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        param = pybamm.ParameterValues("MSMR_Example")
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.build(initial_soc=0.5)
        assert sim._built_initial_soc == 0.5

        # Test whether initial_soc works with half cell composite positive electrode
        # Use actual negative electrode parameters from Chen2020_composite
        options = {"working electrode": "positive", "particle phases": ("1", "2")}
        model = pybamm.lithium_ion.SPM(options)
        param = pybamm.ParameterValues("Chen2020_composite")

        # Map Chen2020_composite negative electrode parameters to positive electrode
        # Primary phase: Graphite (from Chen2020_composite negative primary)
        # Secondary phase: Silicon (from Chen2020_composite negative secondary)

        param.update(
            {
                # Primary phase (Graphite-like from Chen2020_composite negative)
                "Primary: Maximum concentration in positive electrode [mol.m-3]": (
                    param[
                        "Primary: Maximum concentration in negative electrode [mol.m-3]"
                    ]
                ),
                "Primary: Initial concentration in positive electrode [mol.m-3]": (
                    param[
                        "Primary: Initial concentration in negative electrode [mol.m-3]"
                    ]
                ),
                "Primary: Positive particle diffusivity [m2.s-1]": (
                    param["Primary: Negative particle diffusivity [m2.s-1]"]
                ),
                "Primary: Positive electrode OCP [V]": (
                    param["Primary: Negative electrode OCP [V]"]
                ),
                "Primary: Positive electrode active material volume fraction": (
                    param["Primary: Negative electrode active material volume fraction"]
                ),
                "Primary: Positive particle radius [m]": (
                    param["Primary: Negative particle radius [m]"]
                ),
                "Primary: Positive electrode exchange-current density [A.m-2]": (
                    param[
                        "Primary: Negative electrode exchange-current density [A.m-2]"
                    ]
                ),
                "Primary: Positive electrode density [kg.m-3]": (
                    param["Primary: Negative electrode density [kg.m-3]"]
                ),
                "Primary: Positive electrode OCP entropic change [V.K-1]": (
                    param["Primary: Negative electrode OCP entropic change [V.K-1]"]
                ),
                # Secondary phase (Silicon-like from Chen2020_composite negative)
                "Secondary: Maximum concentration in positive electrode [mol.m-3]": (
                    param[
                        "Secondary: Maximum concentration in negative electrode [mol.m-3]"
                    ]
                ),
                "Secondary: Initial concentration in positive electrode [mol.m-3]": (
                    param[
                        "Secondary: Initial concentration in negative electrode [mol.m-3]"
                    ]
                ),
                "Secondary: Positive particle diffusivity [m2.s-1]": (
                    param["Secondary: Negative particle diffusivity [m2.s-1]"]
                ),
                "Secondary: Positive electrode lithiation OCP [V]": (
                    param["Secondary: Negative electrode lithiation OCP [V]"]
                ),
                "Secondary: Positive electrode delithiation OCP [V]": (
                    param["Secondary: Negative electrode delithiation OCP [V]"]
                ),
                "Secondary: Positive electrode OCP [V]": (
                    param["Primary: Negative electrode OCP [V]"]
                ),
                "Secondary: Positive electrode active material volume fraction": (
                    param[
                        "Secondary: Negative electrode active material volume fraction"
                    ]
                ),
                "Secondary: Positive particle radius [m]": (
                    param["Secondary: Negative particle radius [m]"]
                ),
                "Secondary: Positive electrode exchange-current density [A.m-2]": (
                    param[
                        "Secondary: Negative electrode exchange-current density [A.m-2]"
                    ]
                ),
                "Secondary: Positive electrode density [kg.m-3]": (
                    param["Secondary: Negative electrode density [kg.m-3]"]
                ),
                "Secondary: Positive electrode OCP entropic change [V.K-1]": (
                    param["Secondary: Negative electrode OCP entropic change [V.K-1]"]
                ),
            },
            check_already_exists=False,
        )

        # Set voltage cutoffs to match the graphite/silicon OCP range
        param["Lower voltage cut-off [V]"] = (
            0.1  # Very small number < 1e-2 as requested
        )
        param["Upper voltage cut-off [V]"] = 1.0  # Matches silicon OCP range
        param["Open-circuit voltage at 0% SOC [V]"] = 0.1
        param["Open-circuit voltage at 100% SOC [V]"] = 1.0

        # Keep other positive electrode parameters the same
        param["Positive electrode conductivity [S.m-1]"] = param[
            "Positive electrode conductivity [S.m-1]"
        ]
        param["Positive electrode porosity"] = param["Positive electrode porosity"]
        param["Positive electrode Bruggeman coefficient (electrolyte)"] = param[
            "Positive electrode Bruggeman coefficient (electrolyte)"
        ]
        param["Positive electrode Bruggeman coefficient (electrode)"] = param[
            "Positive electrode Bruggeman coefficient (electrode)"
        ]
        param["Positive electrode charge transfer coefficient"] = param[
            "Positive electrode charge transfer coefficient"
        ]
        param["Positive electrode double-layer capacity [F.m-2]"] = param[
            "Positive electrode double-layer capacity [F.m-2]"
        ]
        param["Positive electrode specific heat capacity [J.kg-1.K-1]"] = param[
            "Positive electrode specific heat capacity [J.kg-1.K-1]"
        ]
        param["Positive electrode thermal conductivity [W.m-1.K-1]"] = param[
            "Positive electrode thermal conductivity [W.m-1.K-1]"
        ]

        # Add lithium metal electrode parameters required for half-cell
        # Import the lithium metal exchange current density function
        def li_metal_electrolyte_exchange_current_density_Xu2019(c_e, c_Li, T):
            """
            Exchange-current density for Butler-Volmer reactions between li metal and LiPF6 in
            EC:DMC.
            """
            import pybamm

            m_ref = (
                3.5e-8 * pybamm.constants.F
            )  # (A/m2)(mol/m3) - includes ref concentrations
            return m_ref * c_Li**0.7 * c_e**0.3

        param.update(
            {
                "Exchange-current density for lithium metal electrode [A.m-2]": li_metal_electrolyte_exchange_current_density_Xu2019,
                "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,  # From Xu2019 parameter set
                "Lithium metal interface surface potential difference [V]": 0.0,
                "Current function [A]": 0.0,
            },
            check_already_exists=False,
        )

        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve([0, 1], initial_soc=0.8)
        assert sim._built_initial_soc == 0.8

        # Test with initial voltage for composite half-cell
        sim = pybamm.Simulation(model, parameter_values=param)
        sol = sim.solve(
            [0, 1], initial_soc="0.15 V"
        )  # Test with voltage initialization within composite OCP range
        voltage = sol["Terminal voltage [V]"].entries
        assert voltage[0] == pytest.approx(
            0.15, abs=1e-5
        )  # More relaxed tolerance for composite electrode initialization

        with pytest.warns(DeprecationWarning):
            sim.set_initial_soc(0.5, None)

    def test_solve_with_initial_soc_with_input_param_in_ocv(self):
        # test having an input parameter in the ocv function
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        a = pybamm.Parameter("a")

        def ocv_with_parameter(sto):
            u_eq = (4.2 - 2.5) * (1 - sto) + 2.5
            return a * u_eq

        parameter_values.update(
            {
                "Positive electrode OCP [V]": ocv_with_parameter,
            }
        )
        parameter_values.update({"a": "[input]"}, check_already_exists=False)
        experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        sim.solve([0, 3600], inputs={"a": 1}, initial_soc=0.8)
        assert sim._built_initial_soc == 0.8

    def test_restricted_input_params(self):
        model = pybamm.lithium_ion.SPM()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Initial temperature [K]": "[input]"})
        experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])
        sim = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        )
        with pytest.raises(pybamm.ModelError, match=r"Initial temperature"):
            sim.solve([0, 3600])

    def test_esoh_with_input_param(self):
        # Test that initial soc works with a relevant input parameter
        model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
        param = model.default_parameter_values
        original_eps_p = param["Positive electrode active material volume fraction"]
        param["Positive electrode active material volume fraction"] = (
            pybamm.InputParameter("eps_p")
        )
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(t_eval=[0, 1], initial_soc=0.8, inputs={"eps_p": original_eps_p})
        assert sim._built_initial_soc == 0.8

    def test_solve_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.solve(t_eval=[0, 600], inputs={"Current function [A]": 1})
        np.testing.assert_array_equal(
            sim.solution.all_inputs[0]["Current function [A]"], 1
        )

    def test_solve_with_sensitivities(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        h = 1e-6
        tmax = 600
        t_interp = np.linspace(0, tmax, 100)
        sol1 = sim.solve(
            t_eval=[0, tmax],
            t_interp=t_interp,
            inputs={"Current function [A]": 1},
            calculate_sensitivities=True,
        )

        # check that the sensitivities are stored
        assert "Current function [A]" in sol1.sensitivities

        sol2 = sim.solve(
            t_eval=[0, tmax], t_interp=t_interp, inputs={"Current function [A]": 1 + h}
        )

        # check that the sensitivities are not stored
        assert "Current function [A]" not in sol2.sensitivities

        # check that the sensitivities are roughly correct
        np.testing.assert_allclose(
            sol1["Terminal voltage [V]"].entries
            + h
            * sol1["Terminal voltage [V]"]
            .sensitivities["Current function [A]"]
            .flatten(),
            sol2["Terminal voltage [V]"].entries,
            rtol=5e-6,
            atol=2e-5,
        )

    def test_step_with_inputs(self):
        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.step(
            dt, inputs={"Current function [A]": 1}
        )  # 1 step stores first 12 points
        assert sim.solution.t.size == 12
        assert sim.solution.y[0, :].size == 12
        assert sim.solution.t[0] == 0
        assert sim.solution.t[-1] == dt
        np.testing.assert_array_equal(
            sim.solution.all_inputs[0]["Current function [A]"], 1
        )
        sim.step(
            dt, inputs={"Current function [A]": 2}
        )  # automatically append the next step
        assert sim.solution.y[0, :].size == 24
        np.testing.assert_allclose(
            [sim.solution.t[0], sim.solution.t[-1]],
            np.array([0, 2 * dt]),
            rtol=1e-7,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            sim.solution.all_inputs[1]["Current function [A]"], 2
        )

    def test_time_varying_input_function(self):
        tf = 20.0

        def oscillating(t):
            return 3.6 + 0.1 * np.sin(2 * np.pi * t / tf)

        model = pybamm.lithium_ion.SPM()

        operating_modes = {
            "Current [A]": pybamm.step.current,
            "C-rate": pybamm.step.c_rate,
            "Voltage [V]": pybamm.step.voltage,
            "Power [W]": pybamm.step.power,
        }
        for name in operating_modes:
            operating_mode = operating_modes[name]
            step = operating_mode(oscillating, duration=tf / 2)
            experiment = pybamm.Experiment([step, step], period=f"{tf / 100} seconds")

            solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
            sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
            sim.solve()
            for sol in sim.solution.sub_solutions:
                t0 = sol.t[0]
                np.testing.assert_allclose(
                    sol[name].entries,
                    np.array(oscillating(sol.t - t0)),
                    rtol=1e-7,
                    atol=1e-6,
                )

            # check improper inputs
            for x in (np.nan, np.inf):

                def f(t, x=x):
                    return x + t

                with pytest.raises(ValueError):
                    operating_mode(f)

            def g(t, y):
                return t

            with pytest.raises(TypeError):
                operating_mode(g)

    def test_save_load(self, tmp_path):
        test_name = tmp_path / "tests.pickle"

        model = pybamm.lead_acid.LOQS()
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        sim.save(test_name)
        sim_load = pybamm.load_sim(test_name)
        assert sim.model.name == sim_load.model.name

        # Save after solving
        sim.solve([0, 600])
        sim.save(test_name)
        sim_load = pybamm.load_sim(test_name)
        assert sim.model.name == sim_load.model.name

        # with python formats
        model.convert_to_format = None
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])
        sim.save(test_name)
        model.convert_to_format = "python"
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])
        with pytest.raises(
            NotImplementedError,
            match=r"Cannot save simulation if model format is python",
        ):
            sim.save(test_name)

    def test_load_param(self, tmp_path):
        filename = str(tmp_path / "test.pkl")
        model = pybamm.lithium_ion.SPM()
        params = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=params)
        sim.solve([0, 3600])
        sim.save(filename)
        pkl_obj = pybamm.load_sim(filename)
        assert (
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020"
            == pkl_obj.parameter_values[
                "Negative electrode exchange-current density [A.m-2]"
            ].__name__
        )

    def test_save_load_dae(self, tmp_path):
        test_name = tmp_path / "test.pickle"

        model = pybamm.lead_acid.LOQS({"surface form": "algebraic"})
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        # save after solving
        sim.solve([0, 600])
        sim.save(test_name)
        sim_load = pybamm.load_sim(test_name)
        assert sim.model.name == sim_load.model.name

        # with python format
        model.convert_to_format = None
        sim = pybamm.Simulation(model)
        sim.solve([0, 600])
        sim.save(test_name)

        # with Casadi format & experiment
        model.convert_to_format = "casadi"
        sim = pybamm.Simulation(
            model,
            experiment="Discharge at 1C for 20 minutes",
        )
        sim.solve([0, 600])
        sim.save(test_name)
        sim_load = pybamm.load_sim(test_name)
        assert sim.model.name == sim_load.model.name

    def test_save_load_model(self):
        model = pybamm.lead_acid.LOQS({"surface form": "algebraic"})
        model.use_jacobian = True
        sim = pybamm.Simulation(model)

        # test exception if not discretised
        with pytest.raises(NotImplementedError):
            sim.save_model("sim_save")

        # save after solving
        sim.solve([0, 600])
        sim.save_model("sim_save")

        # load model
        saved_model = pybamm.load_model("sim_save.json")

        assert model.options == saved_model.options

        os.remove("sim_save.json")

    def test_save_load_outvars(self, tmp_path):
        filename = str(tmp_path / "test.pkl")
        model = pybamm.lithium_ion.SPM()
        solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(model, solver=solver)
        sim.solve([0, 600])
        sim.save(filename)
        pkl_obj = pybamm.load_sim(filename)
        assert list(pkl_obj.solver.output_variables) == ["Voltage [V]"]

    def test_plot(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        # test exception if not solved
        with pytest.raises(ValueError):
            sim.plot()

        # now solve and plot
        t_eval = np.linspace(0, 100, 5)
        sim.solve(t_eval=t_eval)
        sim.plot(show_plot=False)

    def test_create_gif(self, tmp_path):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
        with pytest.raises(
            ValueError, match=r"The simulation has not been solved yet."
        ):
            sim.create_gif()
        sim.solve(t_eval=[0, 10])

        # Create a temporary file name
        test_file = tmp_path / "test_sim.gif"

        # create a GIF without calling the plot method
        sim.create_gif(number_of_images=3, duration=1, output_filename=test_file)

        # call the plot method before creating the GIF
        sim.plot(show_plot=False)
        sim.create_gif(number_of_images=3, duration=1, output_filename=test_file)

    @pytest.mark.skipif(
        no_internet_connection(),
        reason="Network not available to download files from registry",
    )
    def test_drive_cycle_interpolant(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        # Import drive cycle from file
        data_loader = pybamm.DataLoader()
        drive_cycle = pd.read_csv(
            pybamm.get_parameters_filepath(data_loader.get_data("US06.csv")),
            comment="#",
            skip_blank_lines=True,
            header=None,
        ).to_numpy()

        current_interpolant = pybamm.Interpolant(
            drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t
        )

        param["Current function [A]"] = current_interpolant

        time_data = drive_cycle[:, 0]

        sim = pybamm.Simulation(model, parameter_values=param)

        # check solution is returned at the times in the data
        sim.solve()
        i = 0
        for t in time_data:
            while abs(sim.solution.t[i] - t) < 1e-6:
                i += 1
                assert i < len(sim.solution.t)

    # Test with an ODE and DAE model
    @pytest.mark.parametrize(
        "model", [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.DFN()]
    )
    def test_heaviside_current(self, model):
        def car_current(t):
            current = (
                1 * (t <= 1000)
                - 0.5 * (1000 < t) * (t < 1500)
                + 0.5 * (2000 < t)
                + 5 * (t >= 3601)
            )
            return current

        def prevfloat(t):
            return np.nextafter(np.float64(t), -np.inf)

        def nextfloat(t):
            return np.nextafter(np.float64(t), np.inf)

        t_eval = [0.0, 3600.0]

        t_nodes = np.array(
            [
                0.0,  # t_eval[0]
                1000.0,  # t <= 1000
                nextfloat(1000.0),  # t <= 1000
                prevfloat(1500.0),  # t < 1500
                1500.0,  # t < 1500
                2000.0,  # 2000 < t
                nextfloat(2000.0),  # 2000 < t
                3600.0,  # t_eval[-1]
            ]
        )

        param = model.default_parameter_values
        param["Current function [A]"] = car_current

        sim = pybamm.Simulation(model, parameter_values=param)

        # Set t_interp to t_eval to only return the breakpoints
        sol = sim.solve(t_eval, t_interp=t_eval)

        np.testing.assert_array_equal(sol.t, t_nodes)
        # Make sure t_eval is not modified
        assert t_eval == [0.0, 3600.0]

        current = sim.solution["Current [A]"]

        for t_node in t_nodes:
            assert current(t_node) == pytest.approx(car_current(t_node))

    # Test with an ODE and DAE model
    @pytest.mark.parametrize(
        "model", [pybamm.lithium_ion.SPM(), pybamm.lithium_ion.DFN()]
    )
    def test_modulo_current(self, model):
        dt = 1.0

        def sawtooth_current(t):
            return t % dt

        def prevfloat(t):
            return np.nextafter(np.float64(t), -np.inf)

        t_eval = [0.0, 10.5]

        t_nodes = np.arange(0.0, 10.5 + dt, dt)
        t_nodes = np.concatenate(
            [
                t_nodes,
                prevfloat(t_nodes),
                t_eval,
            ]
        )

        # Filter out all points not within t_eval
        t_nodes = t_nodes[(t_nodes >= t_eval[0]) & (t_nodes <= t_eval[1])]

        t_nodes = np.sort(np.unique(t_nodes))

        param = model.default_parameter_values
        param["Current function [A]"] = sawtooth_current

        sim = pybamm.Simulation(model, parameter_values=param)

        # Set t_interp to t_eval to only return the breakpoints
        sol = sim.solve(t_eval, t_interp=t_eval)

        np.testing.assert_array_equal(sol.t, t_nodes)
        # Make sure t_eval is not modified
        assert t_eval == [0.0, 10.5]

        current = sim.solution["Current [A]"]

        for t_node in t_nodes:
            assert current(t_node) == pytest.approx(sawtooth_current(t_node))

    def test_filter_discontinuities_simple(self):
        t_eval = [0.0, 3.0, 10.0]
        t_discon = [-5.0, 0.0, 1.0, 3.0, 3.0, 5.0, 10.0, 12.0]

        result = BaseSolver.filter_discontinuities(t_discon, t_eval)
        expected = np.array([1.0, 3.0, 5.0])

        # Exclusive of endpoints
        t_eval_endpoints = [t_eval[0], t_eval[-1]]
        assert all(t not in result for t in t_eval_endpoints)

        np.testing.assert_array_equal(result, expected)

    def test_t_eval(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        # test no t_eval
        with pytest.raises(pybamm.SolverError, match=r"'t_eval' must be provided"):
            sim.solve()

        # test t_eval list of length != 2
        with pytest.raises(pybamm.SolverError, match=r"'t_eval' can be provided"):
            sim.solve(t_eval=[0, 1, 2])

    def test_battery_model_with_input_height(self):
        parameter_values = pybamm.ParameterValues("Marquis2019")
        model = pybamm.lithium_ion.SPM()
        parameter_values.update({"Electrode height [m]": "[input]"})
        # solve model for 1 minute
        t_eval = np.linspace(0, 60, 11)
        inputs = {"Electrode height [m]": 0.2}
        sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
        sim.solve(t_eval=t_eval, inputs=inputs)

    def test_simulation_cannot_force_calc_esoh(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        sim = pybamm.Simulation(model)

        with pytest.warns(
            UserWarning, match=r"Model is not suitable for calculating eSOH"
        ):
            sim.solve([0, 1], calc_esoh=True)
