#
# Tests for the Solution class
#
import io
import json
import logging

import numpy as np
import pandas as pd
import pytest
import scipy
from scipy.io import loadmat

import pybamm
from tests import get_discretisation_for_testing


class TestSolution:
    def test_init(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.Solution(t, y, pybamm.BaseModel(), {})
        np.testing.assert_array_equal(sol.t, t)
        np.testing.assert_array_equal(sol.y, y)
        assert sol.t_event is None
        assert sol.y_event is None
        assert sol.termination == "final time"
        assert sol.all_inputs == [{}]
        assert isinstance(sol.all_models[0], pybamm.BaseModel)

    def test_yp(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        yp = np.tile(t, (20, 1)) * 2  # time derivatives

        # Without yps, yp should be None
        sol_no_yp = pybamm.Solution(t, y, pybamm.BaseModel(), {})
        assert sol_no_yp.hermite_interpolation is False
        assert sol_no_yp.yp is None

        # With yps, yp should return the concatenated time derivatives
        sol_with_yp = pybamm.Solution(t, y, pybamm.BaseModel(), {}, all_yps=yp)
        assert sol_with_yp.hermite_interpolation is True
        np.testing.assert_array_equal(sol_with_yp.yp, yp)

    def test_sensitivities(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        with pytest.raises(TypeError):
            pybamm.Solution(t, y, pybamm.BaseModel(), {}, sensitivities=1.0)

    def test_errors(self):
        bad_ts = [np.array([1, 2, 3]), np.array([3, 4, 5])]
        sol = pybamm.Solution(
            bad_ts, [np.ones((1, 3)), np.ones((1, 3))], pybamm.BaseModel(), {}
        )
        with pytest.raises(
            ValueError, match=r"Solution time vector must be strictly increasing"
        ):
            sol.set_t()

        # Create a mock solution with an SPM
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        t = [0, 1]
        sol = sim.solve(t, t_interp=t)

        ts = sol.all_ts[0]
        bad_ys = np.full_like(sol.all_ys[0], pybamm.settings.max_y_value + 1)
        model = sol.all_models[0]

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.ERROR)
        logger = logging.getLogger("pybamm.logger")
        logger.addHandler(handler)
        pybamm.Solution(ts, bad_ys, model, {})
        log_output = log_capture.getvalue()
        assert "exceeds the maximum" in log_output
        logger.removeHandler(handler)

        with pytest.raises(TypeError, match=r"sensitivities arg needs to be a dict"):
            pybamm.Solution(ts, bad_ys, model, {}, all_sensitivities="bad")

        sol = pybamm.Solution(ts, bad_ys, model, {}, all_sensitivities={})

    def test_add_solutions(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        yp1 = np.tile(t1, (30, 1))
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), {"a": 1}, all_yps=yp1)
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (20, 1))
        yp2 = np.tile(t1, (30, 1))
        sol2 = pybamm.Solution(t2, y2, pybamm.BaseModel(), {"a": 2}, all_yps=yp2)
        sol2.solve_time = 1
        sol2.integration_time = 0.5

        sol_sum = sol1 + sol2

        # Test
        assert sol_sum.integration_time == 0.8
        np.testing.assert_array_equal(sol_sum.t, np.concatenate([t1, t2[1:]]))
        np.testing.assert_array_equal(
            sol_sum.y, np.concatenate([y1, y2[:, 1:]], axis=1)
        )
        np.testing.assert_array_equal(sol_sum.all_inputs, [{"a": 1}, {"a": 2}])

        # Test sub-solutions
        assert len(sol_sum.sub_solutions) == 2
        np.testing.assert_array_equal(sol_sum.sub_solutions[0].t, t1)
        np.testing.assert_array_equal(sol_sum.sub_solutions[1].t, t2)
        assert sol_sum.sub_solutions[0].all_models[0] == sol_sum.all_models[0]
        np.testing.assert_array_equal(sol_sum.sub_solutions[0].all_inputs[0]["a"], 1)
        assert sol_sum.sub_solutions[1].all_models[0] == sol2.all_models[0]
        assert sol_sum.all_models[1] == sol2.all_models[0]
        np.testing.assert_array_equal(sol_sum.sub_solutions[1].all_inputs[0]["a"], 2)

        # Add solution already contained in existing solution
        t3 = np.array([2])
        y3 = np.ones((1, 1))
        sol3 = pybamm.Solution(t3, y3, pybamm.BaseModel(), {"a": 3})
        assert (sol_sum + sol3).all_ts == sol_sum.copy().all_ts

        # add None
        sol4 = sol3 + None
        assert sol3.all_ys == sol4.all_ys

        # radd
        sol5 = None + sol3
        assert sol3.all_ys == sol5.all_ys

        # radd failure
        with pytest.raises(
            pybamm.SolverError,
            match=r"Only a Solution or None can be added to a Solution",
        ):
            sol3 + 2
        with pytest.raises(
            pybamm.SolverError,
            match=r"Only a Solution or None can be added to a Solution",
        ):
            2 + sol3

        sol1 = pybamm.Solution(
            t1,
            y1,
            pybamm.BaseModel(),
            {},
            all_sensitivities={"test": [np.ones((1, 3))]},
        )
        sol1 = pybamm.Solution(t1, y3, pybamm.BaseModel(), {})
        sol2 = pybamm.Solution(t3, y3, pybamm.BaseModel(), {}, all_sensitivities={})
        sol3 = sol1 + sol2
        assert not sol3._all_sensitivities

    def test_add_solutions_different_models(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), {"a": 1})
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (10, 1))
        sol2 = pybamm.Solution(t2, y2, pybamm.BaseModel(), {"a": 2})
        sol2.solve_time = 1
        sol2.integration_time = 0.5
        sol_sum = sol1 + sol2

        # Test
        np.testing.assert_array_equal(sol_sum.t, np.concatenate([t1, t2[1:]]))
        with pytest.raises(
            pybamm.SolverError, match=r"The solution is made up from different models"
        ):
            sol_sum.y

    def test_add_solutions_with_computed_variables(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        model.variables = {"2u": 2 * u}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Set up first solution
        t1 = np.linspace(0, 1, 50)
        solver = pybamm.IDAKLUSolver(output_variables=["2u"])

        sol1 = solver.solve(model, t1)

        # second solution
        t2 = np.linspace(2, 3, 50)
        sol2 = solver.solve(model, t2)

        sol_sum = sol1 + sol2

        # check varaibles concat appropriately
        assert sol_sum["2u"].data[0] == sol1["2u"].data[0]
        assert sol_sum["2u"].data[-1] == sol2["2u"].data[-1]
        # Check functions still work
        sol_sum["2u"].unroll()
        # check solution still tagged as 'variables_returned'
        assert sol_sum.variables_returned is True

    def test_copy(self):
        # Set up first solution
        t1 = [np.linspace(0, 1), np.linspace(1, 2, 5)]
        y1 = [np.tile(t1[0], (20, 1)), np.tile(t1[1], (20, 1))]
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), [{"a": 1}, {"a": 2}])

        sol1.set_up_time = 0.5
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        sol_copy = sol1.copy()
        assert sol_copy.all_ts == sol1.all_ts
        for ys_copy, ys1 in zip(sol_copy.all_ys, sol1.all_ys, strict=False):
            np.testing.assert_array_equal(ys_copy, ys1)
        assert sol_copy.all_inputs == sol1.all_inputs
        assert sol_copy.all_inputs_casadi == sol1.all_inputs_casadi
        assert sol_copy.set_up_time == sol1.set_up_time
        assert sol_copy.solve_time == sol1.solve_time
        assert sol_copy.integration_time == sol1.integration_time

    def test_copy_with_computed_variables(self):
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        model.variables = {"2u": 2 * u}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Set up first solution
        t1 = np.linspace(0, 1, 50)
        solver = pybamm.IDAKLUSolver(output_variables=["2u"])

        sol1 = solver.solve(model, t1)

        sol2 = sol1.copy()

        assert (
            sol1._variables[k] == sol2._variables[k] for k in sol1._variables.keys()
        )
        assert sol2.variables_returned is True

    def test_last_state(self):
        # Set up first solution
        t1 = [np.linspace(0, 1), np.linspace(1, 2, 5)]
        y1 = [np.tile(t1[0], (20, 1)), np.tile(t1[1], (20, 1))]
        sol1 = pybamm.Solution(t1, y1, pybamm.BaseModel(), [{"a": 1}, {"a": 2}])

        sol1.set_up_time = 0.5
        sol1.solve_time = 1.5
        sol1.integration_time = 0.3

        sol_last_state = sol1.last_state
        assert sol_last_state.all_ts[0] == 2
        np.testing.assert_array_equal(sol_last_state.all_ys[0], 2)
        assert sol_last_state.all_inputs == sol1.all_inputs[-1:]
        assert sol_last_state.all_inputs_casadi == sol1.all_inputs_casadi[-1:]
        assert sol_last_state.all_models == sol1.all_models[-1:]
        assert sol_last_state.set_up_time == 0
        assert sol_last_state.solve_time == 0
        assert sol_last_state.integration_time == 0

    def test_first_last_state_empty_y(self):
        # check that first and last state work when y is empty
        # due to only variables being returned (required for experiments)
        model = pybamm.BaseModel()
        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: 1 * v}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u: 0, v: 1}
        model.variables = {"2u": 2 * u, "4u": 4 * u}
        model._summary_variables = {"4u": model.variables["4u"]}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Set up first solution
        t1 = np.linspace(0, 1, 50)
        solver = pybamm.IDAKLUSolver(output_variables=["2u"])

        sol1 = solver.solve(model, t1)

        np.testing.assert_array_equal(
            sol1.first_state.all_ys[0], np.array([[0.0], [1.0]])
        )
        # check summay variables not in the solve can be evaluated at the final timestep
        # via 'last_state
        np.testing.assert_allclose(
            sol1.last_state["4u"].entries, np.array([4.0]), rtol=1e-7, atol=1e-6
        )

    def test_cycles(self):
        model = pybamm.lithium_ion.SPM()
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/20 for 0.5 hours", "Charge at C/20 for 15 minutes"),
                ("Discharge at C/20 for 0.5 hours", "Charge at C/20 for 15 minutes"),
            ]
        )
        sim = pybamm.Simulation(model, experiment=experiment)
        sol = sim.solve()
        assert len(sol.cycles) == 2
        len_cycle_1 = len(sol.cycles[0].t)

        assert isinstance(sol.cycles[0], pybamm.Solution)
        np.testing.assert_array_equal(sol.cycles[0].t, sol.t[:len_cycle_1])
        np.testing.assert_array_equal(sol.cycles[0].y, sol.y[:, :len_cycle_1])

        assert isinstance(sol.cycles[1], pybamm.Solution)
        np.testing.assert_array_equal(sol.cycles[1].t, sol.t[len_cycle_1:])
        np.testing.assert_allclose(sol.cycles[1].y, sol.y[:, len_cycle_1:])

    def test_total_time(self):
        sol = pybamm.Solution(np.array([0]), np.array([[1, 2]]), pybamm.BaseModel(), {})
        sol.set_up_time = 0.5
        sol.solve_time = 1.2
        assert sol.total_time == 1.7

    def test_getitem(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c

        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        # test create a new processed variable
        c_sol = solution["c"]
        assert isinstance(c_sol, pybamm.ProcessedVariable)
        np.testing.assert_array_equal(c_sol.entries, c_sol(solution.t))

        # test call an already created variable
        solution.update("2c")
        twoc_sol = solution["2c"]
        assert isinstance(twoc_sol, pybamm.ProcessedVariable)
        np.testing.assert_array_equal(twoc_sol.entries, twoc_sol(solution.t))
        np.testing.assert_array_equal(twoc_sol.entries, 2 * c_sol.entries)

    def test_plot(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c

        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        solution.plot(["c", "2c"], show_plot=False)

    def test_save(self, tmp_path):
        test_stub = tmp_path / "test"

        model = pybamm.BaseModel()
        # create both 1D and 2D variables
        c = pybamm.Variable("c")
        d = pybamm.Variable("d", domain="negative electrode")
        model.rhs = {c: -c, d: 1}
        model.initial_conditions = {c: 1, d: 2}
        model.variables = {"c": c, "d": d, "2c": 2 * c, "c + d": c + d}

        disc = get_discretisation_for_testing()
        disc.process_model(model)
        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        # test save data
        with pytest.raises(ValueError):
            solution.save_data(f"{test_stub}.pickle")

        # set variables first then save
        solution.update(["c", "d"])
        with pytest.raises(ValueError, match=r"pickle"):
            solution.save_data(to_format="pickle")
        solution.save_data(f"{test_stub}.pickle")

        data_load = pybamm.load(f"{test_stub}.pickle")
        np.testing.assert_array_equal(solution.data["c"], data_load["c"])
        np.testing.assert_array_equal(solution.data["d"], data_load["d"])

        # to matlab
        solution.save_data(f"{test_stub}.mat", to_format="matlab")
        data_load = loadmat(f"{test_stub}.mat")
        np.testing.assert_array_equal(solution.data["c"], data_load["c"].flatten())
        np.testing.assert_array_equal(solution.data["d"], data_load["d"])

        with pytest.raises(ValueError, match=r"matlab"):
            solution.save_data(to_format="matlab")

        # to matlab with bad variables name fails
        solution.update(["c + d"])
        with pytest.raises(ValueError, match=r"Invalid character"):
            solution.save_data(f"{test_stub}.mat", to_format="matlab")
        # Works if providing alternative name
        solution.save_data(
            f"{test_stub}.mat",
            to_format="matlab",
            short_names={"c + d": "c_plus_d"},
        )
        data_load = loadmat(f"{test_stub}.mat")
        np.testing.assert_array_equal(solution.data["c + d"], data_load["c_plus_d"])

        # to csv
        with pytest.raises(ValueError, match=r"only 0D variables can be saved to csv"):
            solution.save_data(f"{test_stub}.csv", to_format="csv")
        # only save "c" and "2c"
        solution.save_data(f"{test_stub}.csv", ["c", "2c"], to_format="csv")
        csv_str = solution.save_data(variables=["c", "2c"], to_format="csv")

        # check string is the same as the file
        with open(f"{test_stub}.csv") as f:
            # need to strip \r chars for windows
            assert csv_str.replace("\r", "") == f.read()

        # read csv
        df = pd.read_csv(f"{test_stub}.csv")
        np.testing.assert_allclose(df["c"], solution.data["c"], rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(df["2c"], solution.data["2c"], rtol=1e-7, atol=1e-6)

        # to json
        solution.save_data(f"{test_stub}.json", to_format="json")
        json_str = solution.save_data(to_format="json")

        # check string is the same as the file
        with open(f"{test_stub}.json") as f:
            # need to strip \r chars for windows
            assert json_str.replace("\r", "") == f.read()

        # check if string has the right values
        json_data = json.loads(json_str)
        np.testing.assert_allclose(
            json_data["c"], solution.data["c"], rtol=1e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            json_data["d"], solution.data["d"], rtol=1e-7, atol=1e-6
        )

        # raise error if format is unknown
        with pytest.raises(ValueError, match=r"format 'wrong_format' not recognised"):
            solution.save_data(f"{test_stub}.csv", to_format="wrong_format")

        # test save whole solution
        solution.save(f"{test_stub}.pickle")
        solution_load = pybamm.load(f"{test_stub}.pickle")
        assert solution.all_models[0].name == solution_load.all_models[0].name
        np.testing.assert_array_equal(solution["c"].entries, solution_load["c"].entries)
        np.testing.assert_array_equal(solution["d"].entries, solution_load["d"].entries)

    def test_get_data_cycles_steps(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c

        solver = pybamm.ScipySolver()
        sol1 = solver.solve(model, np.linspace(0, 1))
        sol2 = solver.solve(model, np.linspace(1, 2))

        sol = sol1 + sol2
        sol.cycles = [sol]
        sol.cycles[0].steps = [sol1, sol2]

        data = sol.get_data_dict("c")
        np.testing.assert_array_equal(data["Cycle"], 0)
        np.testing.assert_array_equal(
            data["Step"], np.concatenate([np.zeros(50), np.ones(50)])
        )

    def test_solution_evals_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({"Negative electrode conductivity [S.m-1]": "[input]"})
        var_pts = {"x_n": 5, "x_s": 5, "x_p": 5, "r_n": 10, "r_p": 10}
        spatial_methods = model.default_spatial_methods
        solver = model.default_solver
        sim = pybamm.Simulation(
            model=model,
            geometry=geometry,
            parameter_values=param,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )
        inputs = {"Negative electrode conductivity [S.m-1]": 0.1}
        sim.solve(t_eval=[0, 10], t_interp=np.linspace(0, 10, 10), inputs=inputs)
        time = sim.solution["Time [h]"](sim.solution.t)
        assert len(time) == 10

    def test_discrete_data_sum_errors(self):
        data_times = np.array([0.0])
        data_values = np.array([1.0])
        data = pybamm.DiscreteTimeData(data_times, data_values, "test_data")
        dts = pybamm.DiscreteTimeSum(data)

        model = pybamm.BaseModel(name="test_model2")
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["dts"] = pybamm.t * dts
        solver = pybamm.IDAKLUSolver()
        with pytest.raises(
            ValueError,
            match=r"time or state vector nodes should only appear within the time integral node",
        ):
            solver.solve(model, t_eval=[0, 0.1])["dts"]

        model = pybamm.BaseModel(name="test_model2")
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["dts"] = dts * dts
        solver = pybamm.IDAKLUSolver()
        with pytest.raises(
            ValueError,
            match=r"More than one time integral node found",
        ):
            solver.solve(model, t_eval=[0, 0.1])["dts"]

    _solver_classes = [
        (pybamm.CasadiSolver, False, False),
        (pybamm.IDAKLUSolver, False, False),
        (pybamm.CasadiSolver, True, False),
        (pybamm.IDAKLUSolver, True, False),
        (pybamm.IDAKLUSolver, False, True),
        (pybamm.IDAKLUSolver, True, True),
    ]

    @pytest.mark.parametrize(
        "solver_class,use_post_sum,use_output_var", _solver_classes
    )
    def test_discrete_data_sum(self, solver_class, use_post_sum, use_output_var):
        model = pybamm.BaseModel(name="test_model")
        c = pybamm.Variable("c")
        model.rhs = {c: -2 * c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c

        data_times = np.linspace(0, 1, 10)
        if solver_class == pybamm.IDAKLUSolver:
            t_eval = [data_times[0], data_times[-1]]
            t_interp = data_times
        else:
            t_eval = data_times
            t_interp = None
        solver = solver_class()
        data_values = solver.solve(model, t_eval=t_eval, t_interp=t_interp)["c"].entries

        data = pybamm.DiscreteTimeData(data_times, data_values, "test_data")
        if use_post_sum:
            data_comparison = (pybamm.DiscreteTimeSum((c - data) ** 2)) ** 0.5
        else:
            data_comparison = pybamm.DiscreteTimeSum((c - data) ** 2)

        model = pybamm.BaseModel(name="test_model2")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        c2 = pybamm.Variable("c2")
        model.rhs = {c: b * -a * c, c2: -2 * c2}
        model.initial_conditions = {c: 1, c2: 1}
        model.variables["data_comparison"] = data_comparison
        model.variables["data"] = data
        model.variables["c"] = c

        if use_output_var:
            output_variables = ["data_comparison", "c", "data"]
            solver = solver_class(output_variables=output_variables)
        else:
            solver = solver_class()
        range = [0.5, 1.0, 2.0]
        range2 = np.ones(3)
        for a, b in zip(range, range2, strict=False):
            sol = solver.solve(
                model, t_eval=t_eval, t_interp=t_interp, inputs={"a": a, "b": b}
            )
            y_sol = np.exp(b * -a * data_times)
            if use_post_sum:
                expected = np.sqrt(np.sum((y_sol - data_values) ** 2))
            else:
                expected = np.sum((y_sol - data_values) ** 2)
            np.testing.assert_allclose(
                sol["data_comparison"](), expected, rtol=1e-3, atol=1e-2
            )
            assert isinstance(sol["data_comparison"].data, np.ndarray)
            assert sol["data_comparison"].data.shape == (1,)

            # sensitivity calculation only supported for IDAKLUSolver
            if solver_class == pybamm.IDAKLUSolver:
                sol = solver.solve(
                    model,
                    t_eval=t_eval,
                    t_interp=t_interp,
                    inputs={"a": a, "b": b},
                    calculate_sensitivities=True,
                )
                y_sol = np.exp(b * -a * data_times)
                dy_sol_da = -data_times * y_sol
                if use_post_sum:
                    expected_sens = (
                        0.5
                        * (expected ** (-0.5))
                        * np.sum(2 * (y_sol - data_values) * dy_sol_da)
                    )
                else:
                    expected_sens = np.sum(2 * (y_sol - data_values) * dy_sol_da)

                np.testing.assert_allclose(
                    sol["data"].sensitivities["a"].flatten(),
                    np.zeros_like(data_times),
                    rtol=1e-3,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    sol["c"].data,
                    y_sol,
                    rtol=1e-3,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    sol["c"].sensitivities["a"].flatten(),
                    dy_sol_da,
                    rtol=1e-3,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    sol["data_comparison"].sensitivities["a"],
                    expected_sens,
                    rtol=1e-3,
                    atol=1e-2,
                )
                assert isinstance(sol["data_comparison"].sensitivities["a"], np.ndarray)
                assert sol["data_comparison"].sensitivities["a"].shape == (1,)

                # should raise error if t_interp is not equal to data_times
                with pytest.raises(
                    pybamm.SolverError,
                    match=r"solution times and discrete times of the time integral are not equal",
                ):
                    solver.solve(
                        model,
                        t_eval=t_eval,
                        inputs={"a": a, "b": b},
                        calculate_sensitivities=True,
                    )["data_comparison"].sensitivities["a"]

    @pytest.mark.parametrize(
        "solver_class,use_post_sum,use_output_var", _solver_classes
    )
    def test_explicit_time_integral(self, solver_class, use_post_sum, use_output_var):
        times = np.linspace(0, 1, 10)
        c = pybamm.Variable("c")
        if solver_class == pybamm.IDAKLUSolver:
            t_eval = [times[0], times[-1]]
            t_interp = times
        else:
            t_eval = times
            t_interp = None

        if use_post_sum:
            integral = pybamm.ExplicitTimeIntegral(c, 0) ** 2
        else:
            integral = pybamm.ExplicitTimeIntegral(c, 0)

        model = pybamm.BaseModel(name="test_model")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        c2 = pybamm.Variable("c2")
        model.rhs = {c: b * -a * c, c2: -2 * c2}
        model.initial_conditions = {c: 1, c2: 1}
        model.variables["integral"] = integral
        model.variables["c"] = c

        if use_output_var:
            output_variables = ["integral", "c"]
            solver = solver_class(output_variables=output_variables)
        else:
            solver = solver_class()
        range = [0.5, 1.0, 2.0]
        range2 = np.ones(3)
        for a, b in zip(range, range2, strict=False):
            sol = solver.solve(
                model, t_eval=t_eval, t_interp=t_interp, inputs={"a": a, "b": b}
            )
            y_sol = np.exp(b * -a * times)
            expected = -(1.0 / b / a) * (
                np.exp(b * -a * times[-1]) - np.exp(b * -a * times[0])
            )
            if use_post_sum:
                expected = expected**2
            np.testing.assert_allclose(
                sol["integral"](), expected, rtol=1e-3, atol=1e-2
            )
            assert isinstance(sol["integral"].data, np.ndarray)

            # sensitivity calculation only supported for IDAKLUSolver
            if solver_class == pybamm.IDAKLUSolver:
                sol = solver.solve(
                    model,
                    t_eval=t_eval,
                    t_interp=t_interp,
                    inputs={"a": a, "b": b},
                    calculate_sensitivities=True,
                )
                y_sol = np.exp(b * -a * times)
                dy_sol_da = -b * times * y_sol
                expected_sens = scipy.integrate.trapezoid(dy_sol_da, times)
                if use_post_sum:
                    expected_sens = 2 * expected * expected_sens

                np.testing.assert_allclose(
                    sol["c"].data,
                    y_sol,
                    rtol=1e-3,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    sol["c"].sensitivities["a"].flatten(),
                    dy_sol_da,
                    rtol=1e-3,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    sol["integral"].sensitivities["a"],
                    expected_sens,
                    rtol=1e-3,
                    atol=1e-2,
                )
                assert isinstance(sol["integral"].sensitivities["a"], np.ndarray)

    def test_observe(self):
        """Test the observe method with pybamm symbols, comparing with model variables."""
        # Set up a simple model
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")

        # Solve the model
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve([0, 3600])

        # Test observing "Voltage [V]" symbol - should match exactly with model variable
        voltage_symbol = model.variables["Voltage [V]"]
        observed_voltage = sol.observe(voltage_symbol)

        # Compare with the actual variable from solution
        actual_voltage = sol["Voltage [V]"]

        # They should match exactly
        np.testing.assert_array_equal(observed_voltage.data, actual_voltage.data)
        np.testing.assert_array_equal(observed_voltage.entries, actual_voltage.entries)

        # Test with "Current [A]" - another model variable
        current_symbol = model.variables["Current [A]"]
        observed_current = sol.observe(current_symbol)
        actual_current = sol["Current [A]"]
        np.testing.assert_array_equal(observed_current.data, actual_current.data)
        np.testing.assert_array_equal(observed_current.entries, actual_current.entries)

        # Test that observe returns a ProcessedVariable
        assert isinstance(observed_voltage, pybamm.ProcessedVariable)
        assert isinstance(observed_current, pybamm.ProcessedVariable)

        # Test that we can call observe multiple times and get the same result
        observed_voltage2 = sol.observe(voltage_symbol)
        np.testing.assert_array_equal(observed_voltage2.data, observed_voltage.data)

        # Test that the cache works - verify it's the same object (not just equal)
        observed_voltage3 = sol.observe(voltage_symbol)
        assert observed_voltage3 is observed_voltage  # Should be the same cached object

    def test_observe_with_numeric_inputs(self):
        """Test that observe works with numeric inputs like 0, which get converted to symbols."""
        # Set up a simple model
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)

        sol = sim.solve([0, 1])

        # Test observing a scalar value (0) - should convert to pybamm.Scalar(0)
        observed_zero = sol.observe(0)
        assert isinstance(observed_zero, pybamm.ProcessedVariable)
        # Should be a constant array of zeros
        np.testing.assert_array_equal(observed_zero.data, np.zeros(len(sol.t)))

        # Test that numeric inputs are cached correctly
        observed_zero2 = sol.observe(0)
        assert observed_zero2 is observed_zero  # cached

    def test_observe_failure(self):
        """Test that observe raises an error if the solver includes `output_variables`."""
        # 1. Input is invalid
        t_eval = [0, 1]
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve(t_eval)

        with pytest.raises(ValueError, match=r"Input cannot be converted"):
            sol.observe(None)

        # 2. Trying to observe a symbol which is not part of the parameter_values or model
        symbol = pybamm.Parameter("_not_in_model")
        with pytest.raises(KeyError, match=r"not found"):
            sol.observe(symbol)

        # 3. Solver includes `output_variables` - solution not observable but models
        # can still process symbols
        solver = pybamm.IDAKLUSolver(output_variables=["Voltage [V]"])
        sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)
        sol = sim.solve(t_eval)
        assert sol.observable is False
        # Models can still process symbols (delayed variable processing is enabled)
        assert all(model.can_process_symbols for model in sol.all_models)

        with pytest.raises(ValueError, match=r"solver includes `output_variables`"):
            sol.observe(model.variables["Current [A]"])

        # 4. `disable_solution_observability` is called on the model - solution not
        # observable but models can still process symbols
        model = pybamm.lithium_ion.SPM()
        model.disable_solution_observability(pybamm.ModelSolutionObservability.DISABLED)
        sim = pybamm.Simulation(model)
        sol = sim.solve(t_eval)
        assert sol.observable is False
        assert all(model.can_process_symbols for model in sol.all_models)

        with pytest.raises(ValueError, match=r"disable_solution_observability"):
            sol.observe(model.variables["Current [A]"])

        # 5. Missing non-strictly required input parameters - solution unobservable
        # but models can still process symbols
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        input_names = sorted(
            ["dummy", "Positive electrode active material volume fraction"]
        )
        parameter_values.update(
            {k: "[input]" for k in input_names}, check_already_exists=False
        )
        sim = pybamm.Simulation(model, parameter_values=parameter_values)

        # purposefully missing the dummy input
        inputs = {name: 0.5 for name in input_names if name != "dummy"}

        # check that BaseSolver raises a warning about missing inputs,
        # and is unobservable, but it is still solvable
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("pybamm.logger")
        logger.addHandler(handler)

        sol = sim.solve(t_eval, inputs=inputs)

        log_output = log_capture.getvalue()
        assert "No value provided for input" in log_output
        assert "dummy" in log_output
        assert "can no longer be observed" in log_output
        logger.removeHandler(handler)

        assert sol.observable is False
        assert all(not model.solution_observable for model in sol.all_models)

        model = sol.all_models[0]
        assert set(ip.name for ip in model.input_parameters) == set(input_names)
        assert set(ip.name for ip in model.required_input_parameters) == set(
            inputs.keys()
        )
        # check that missing input is set to DUMMY_INPUT_PARAMETER_VALUE (np.nan)
        assert np.isnan(sol.all_inputs[0]["dummy"])

        with pytest.raises(ValueError, match=r"input parameters were not provided"):
            sol.observe(model.variables["Current [A]"])

        # 6. Model is partially processed before simulation is built - models cannot
        # process symbols at all
        model = pybamm.lithium_ion.SPM()
        parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values.process_model(model)
        sim = pybamm.Simulation(model, parameter_values=parameter_values)
        sol = sim.solve(t_eval)
        assert sol.observable is False
        assert all(not model.can_process_symbols for model in sol.all_models)
        model = sol.all_models[0]

        with pytest.raises(ValueError, match=r"re-parameterised"):
            sol.observe(model.variables["Current [A]"])
