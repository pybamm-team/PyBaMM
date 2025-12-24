#
# Tests for the Solution class
#
import numpy as np

import pybamm
from pybamm.models.base_model import ModelSolutionObservability


class TestSolution:
    def test_append(self):
        model = pybamm.lithium_ion.SPMe()
        # create geometry
        geometry = model.default_geometry

        # load parameter values and process model and geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)

        # set mesh
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # solve model
        t_eval = np.linspace(0, 3600, 100)
        solver = model.default_solver
        solution = solver.solve(model, t_eval)

        # step model
        old_t = 0
        step_solver = model.default_solver
        step_solution = None
        # dt should be dimensional
        for t in solution.t[1:]:
            dt = t - old_t
            step_solution = step_solver.step(step_solution, model, dt=dt, npts=10)
            if t == solution.t[1]:
                # Create voltage variable
                step_solution.update("Voltage [V]")
            old_t = t

        # Check both give the same answer
        np.testing.assert_allclose(
            solution["Voltage [V]"](solution.t[:-1]),
            step_solution["Voltage [V]"](solution.t[:-1]),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_observe(self):
        parameter_values = pybamm.ParameterValues("Chen2020")
        inputs = {
            "dummy": 2.0,
            "Positive electrode active material volume fraction": 0.5,
        }
        t_eval = [0, 1]
        parameter_values.update(
            {k: "[input]" for k in inputs.keys()}, check_already_exists=False
        )

        model_unobservable = pybamm.lithium_ion.SPM()
        model_unobservable.disable_solution_observability(
            ModelSolutionObservability.DISABLED
        )
        sim_unobservable = pybamm.Simulation(
            model_unobservable, parameter_values=parameter_values
        )
        sol_unobservable = sim_unobservable.solve(t_eval, inputs=inputs)
        assert sol_unobservable.observable is False

        model_observable = pybamm.lithium_ion.SPM()
        sim_observable = pybamm.Simulation(
            model_observable, parameter_values=parameter_values
        )
        sol_observable = sim_observable.solve(t_eval, inputs=inputs)
        assert sol_observable.observable is True

        model = pybamm.lithium_ion.SPM()
        for name, variable in model.variables.items():
            out_unobservable = sol_unobservable[name].data
            out_observable = sol_observable.observe(variable).data
            # Check exact equality
            np.testing.assert_array_equal(out_unobservable, out_observable)

        # Check that input parameters which appear in parameter_values but
        # not in the model can still be observed
        def func(current, dummy):
            return current * dummy**2

        symbol = func(model.variables["Current [A]"], pybamm.Parameter("dummy"))
        out_true = func(sol_observable["Current [A]"].data, inputs["dummy"])
        np.testing.assert_allclose(
            sol_observable.observe(symbol).data,
            out_true,
        )
