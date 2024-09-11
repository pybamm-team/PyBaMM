import pybamm
import numpy as np
import pandas as pd

import os
import uuid
import pytest
from tempfile import TemporaryDirectory
from scipy.integrate import cumulative_trapezoid
from tests import no_internet_connection


class TestSimulation:
    def test_simple_model(self):
        model = pybamm.BaseModel()
        v = pybamm.Variable("v")
        model.rhs = {v: -v}
        model.initial_conditions = {v: 1}
        sim = pybamm.Simulation(model)
        sol = sim.solve([0, 1])
        np.testing.assert_array_almost_equal(sol.y.full()[0], np.exp(-sol.t), decimal=5)

    def test_basic_ops(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        # check that the model is unprocessed
        assert sim._mesh == None
        assert sim._disc == None
        V = sim.model.variables["Voltage [V]"]
        assert V.has_symbol_of_classes(pybamm.Parameter)
        assert not V.has_symbol_of_classes(pybamm.Matrix)

        sim.set_parameters()
        assert sim._mesh == None
        assert sim._disc == None
        V = sim.model_with_set_params.variables["Voltage [V]"]
        assert not V.has_symbol_of_classes(pybamm.Parameter)
        assert not V.has_symbol_of_classes(pybamm.Matrix)
        # Make sure model is unchanged
        assert sim.model != model
        V = model.variables["Voltage [V]"]
        assert V.has_symbol_of_classes(pybamm.Parameter)
        assert not V.has_symbol_of_classes(pybamm.Matrix)

        assert sim.submesh_types == model.default_submesh_types
        assert sim.var_pts == model.default_var_pts
        assert sim.mesh is None
        for key in sim.spatial_methods.keys():
            assert (
                sim.spatial_methods[key].__class__
                == model.default_spatial_methods[key].__class__
            )

        sim.build()
        assert sim._mesh is not None
        assert sim._disc is not None
        V = sim.built_model.variables["Voltage [V]"]
        assert not V.has_symbol_of_classes(pybamm.Parameter)
        assert V.has_symbol_of_classes(pybamm.Matrix)

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
        with pytest.raises(ValueError, match="save_at_cycles"):
            sim.solve(save_at_cycles=2)
        with pytest.raises(ValueError, match="starting_solution"):
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
        np.testing.assert_array_almost_equal(
            q, cumulative_trapezoid(I, t, initial=0) / 3600
        )

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
        np.testing.assert_array_almost_equal(
            sim.solution["v"].entries, np.exp(-np.linspace(0, 1, 100))
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

        sim.step(dt)  # 1 step stores first two points
        assert sim.solution.y.full()[0, :].size == 2
        np.testing.assert_array_almost_equal(sim.solution.t, np.array([0, dt]))
        saved_sol = sim.solution

        sim.step(dt)  # automatically append the next step
        assert sim.solution.y.full()[0, :].size == 4
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([0, dt, dt + 1e-9, 2 * dt])
        )

        sim.step(dt, save=False)  # now only store the two end step points
        assert sim.solution.y.full()[0, :].size == 2
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([2 * dt + 1e-9, 3 * dt])
        )
        # Start from saved solution
        sim.step(dt, starting_solution=saved_sol)
        assert sim.solution.y.full()[0, :].size == 4
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([0, dt, dt + 1e-9, 2 * dt])
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
        sol = sim.solve([0, 1], initial_soc=f"{ucv} V")
        voltage = sol["Terminal voltage [V]"].entries
        assert voltage[0] == pytest.approx(ucv, abs=1e-05)

        # test with MSMR
        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        param = pybamm.ParameterValues("MSMR_Example")
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.build(initial_soc=0.5)
        assert sim._built_initial_soc == 0.5

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

    def test_step_with_inputs(self):
        dt = 0.001
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        param.update({"Current function [A]": "[input]"})
        sim = pybamm.Simulation(model, parameter_values=param)
        sim.step(
            dt, inputs={"Current function [A]": 1}
        )  # 1 step stores first two points
        assert sim.solution.t.size == 2
        assert sim.solution.y.full()[0, :].size == 2
        assert sim.solution.t[0] == 0
        assert sim.solution.t[1] == dt
        np.testing.assert_array_equal(
            sim.solution.all_inputs[0]["Current function [A]"], 1
        )
        sim.step(
            dt, inputs={"Current function [A]": 2}
        )  # automatically append the next step
        assert sim.solution.y.full()[0, :].size == 4
        np.testing.assert_array_almost_equal(
            sim.solution.t, np.array([0, dt, dt + 1e-9, 2 * dt])
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

            solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8)
            sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
            sim.solve()
            for sol in sim.solution.sub_solutions:
                t0 = sol.t[0]
                np.testing.assert_array_almost_equal(
                    sol[name].entries, np.array(oscillating(sol.t - t0))
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

    def test_save_load(self):
        with TemporaryDirectory() as dir_name:
            test_name = os.path.join(dir_name, "tests.pickle")

            model = pybamm.lead_acid.LOQS()
            model.use_jacobian = True
            sim = pybamm.Simulation(model)

            sim.save(test_name)
            sim_load = pybamm.load_sim(test_name)
            assert sim.model.name == sim_load.model.name

            # save after solving
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
                match="Cannot save simulation if model format is python",
            ):
                sim.save(test_name)

    def test_load_param(self):
        # Test load_sim for parameters imports
        filename = f"{uuid.uuid4()}.p"
        model = pybamm.lithium_ion.SPM()
        params = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=params)
        sim.solve([0, 3600])
        sim.save(filename)

        try:
            pkl_obj = pybamm.load_sim(os.path.join(filename))
        except Exception as excep:
            os.remove(filename)
            raise excep

        assert (
            "graphite_LGM50_electrolyte_exchange_current_density_Chen2020"
            == pkl_obj.parameter_values[
                "Negative electrode exchange-current density [A.m-2]"
            ].__name__
        )
        os.remove(filename)

    def test_save_load_dae(self):
        with TemporaryDirectory() as dir_name:
            test_name = os.path.join(dir_name, "test.pickle")

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

            # with Casadi solver & experiment
            model.convert_to_format = "casadi"
            sim = pybamm.Simulation(
                model,
                experiment="Discharge at 1C for 20 minutes",
                solver=pybamm.CasadiSolver(),
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

    def test_plot(self):
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM())

        # test exception if not solved
        with pytest.raises(ValueError):
            sim.plot()

        # now solve and plot
        t_eval = np.linspace(0, 100, 5)
        sim.solve(t_eval=t_eval)
        sim.plot(show_plot=False)

    def test_create_gif(self):
        with TemporaryDirectory() as dir_name:
            sim = pybamm.Simulation(pybamm.lithium_ion.SPM())
            with pytest.raises(
                ValueError, match="The simulation has not been solved yet."
            ):
                sim.create_gif()
            sim.solve(t_eval=[0, 10])

            # Create a temporary file name
            test_file = os.path.join(dir_name, "test_sim.gif")

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
        np.testing.assert_array_almost_equal(sim.solution.t, time_data)

        # check warning raised if the largest gap in t_eval is bigger than the
        # smallest gap in the data
        with pytest.warns(pybamm.SolverWarning):
            sim.solve(t_eval=np.linspace(0, 10, 3))

        # check warning raised if t_eval doesnt contain time_data , but has a finer
        # resolution (can still solve, but good for users to know they dont have
        # the solution returned at the data points)
        with pytest.warns(pybamm.SolverWarning):
            sim.solve(t_eval=np.linspace(0, time_data[-1], 800))

    def test_discontinuous_current(self):
        def car_current(t):
            current = (
                1 * (t >= 0) * (t <= 1000)
                - 0.5 * (1000 < t) * (t <= 2000)
                + 0.5 * (2000 < t)
            )
            return current

        model = pybamm.lithium_ion.DFN()
        param = model.default_parameter_values
        param["Current function [A]"] = car_current

        sim = pybamm.Simulation(
            model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
        )
        sim.solve([0, 3600])
        current = sim.solution["Current [A]"]
        assert current(0) == 1
        assert current(1500) == -0.5
        assert current(3000) == 0.5

    def test_t_eval(self):
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        # test no t_eval
        with pytest.raises(pybamm.SolverError, match="'t_eval' must be provided"):
            sim.solve()

        # test t_eval list of length != 2
        with pytest.raises(pybamm.SolverError, match="'t_eval' can be provided"):
            sim.solve(t_eval=[0, 1, 2])

        # tets list gets turned into np.linspace(t0, tf, 100)
        sim.solve(t_eval=[0, 10])
        np.testing.assert_array_almost_equal(sim.solution.t, np.linspace(0, 10, 100))

    def test_battery_model_with_input_height(self):
        parameter_values = pybamm.ParameterValues("Marquis2019")
        model = pybamm.lithium_ion.SPM()
        parameter_values.update({"Electrode height [m]": "[input]"})
        # solve model for 1 minute
        t_eval = np.linspace(0, 60, 11)
        inputs = {"Electrode height [m]": 0.2}
        sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
        sim.solve(t_eval=t_eval, inputs=inputs)
