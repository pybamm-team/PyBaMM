import pybamm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
        
# Spoof mult-XLA device for testing
if True:
    import os
    os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=8"

import jax.numpy as jnp


# Single run toy model

if False:
    model = pybamm.BaseModel('name')
    y = pybamm.Variable('y')
    a = pybamm.Parameter('a')
    model.rhs = {y: a * y}
    model.initial_conditions = {y: 1}
    model.convert_to_format = 'jax'

    print(model.rhs[y])

    params = pybamm.ParameterValues({'a': 2})
    params.process_model(model)
    print(model.rhs[y])

    solver = pybamm.JaxSolver(
        rtol=1e-10, atol=1e-10
    )
    t_eval = np.linspace(0, 1, 80)
    solution = solver.solve(model, t_eval)
    # plt.plot(solution.t, solution.y.T)
    # plt.show()

# Toy model with input parameters

if False:
    model = pybamm.BaseModel('name')
    y = pybamm.Variable('y')
    a = pybamm.InputParameter('a')
    model.rhs = {y: a * y}
    model.initial_conditions = {y: 1}
    model.convert_to_format = 'jax'

    solver = pybamm.JaxSolver(
        rtol=1e-10, atol=1e-10,
        method='BDF'
    )
    t_eval = np.linspace(0, 1, 80)

    # Individually
    if False:
        for a_value in range(1, 5):
            solution = solver.solve(model, t_eval, inputs={'a': a_value})
            plt.plot(solution.t, solution.y.T, label=f'a = {a_value}')
            print(solution.y.T)
        plt.show()

    # Batched
    inputs = [{'a': k} for k in range(1, 8)]
    solution = solver.solve(model, t_eval, inputs=inputs)
    for sol in solution:
        plt.plot(sol.t, sol.y.T)
    plt.show()

# Convert a standard parameter to an input parameter and run

if False:
    model = pybamm.BaseModel('name')
    y = pybamm.Variable('y')
    a = pybamm.Parameter('a')
    model.rhs = {y: a * y}
    model.initial_conditions = {y: 1}
    model.convert_to_format = 'jax'
    model.variables = {"y squared": y ** 2}

    params = pybamm.ParameterValues({'a': '[input]'})
    params.process_model(model)

    solver = pybamm.JaxSolver(
        rtol=1e-10, atol=1e-10
    )
    t_eval = np.linspace(0, 1, 80)
    for a_value in range(1, 5):
        solution = solver.solve(model, t_eval, inputs={'a': a_value})
        plt.plot(solution.t, solution.y.T, label=f'a = {a_value}')
    plt.show()

# Battery model single run with an input parameter

if False:
    model = pybamm.lithium_ion.DFN()
    model.convert_to_format = 'jax'
    model.events = []  # remove events (not supported in jax)
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update({"Current function [A]": "[input]"})
    param.process_geometry(geometry)
    param.process_model(model)

    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    t_eval = np.linspace(0, 3600, 100)
    solver = pybamm.JaxSolver(atol=1e-6, rtol=1e-6, method="BDF")
    solution = solver.solve(
        model,
        t_eval,
        inputs={"Current function [A]": 0.15652}
    )

    plot = pybamm.QuickPlot(
        solution, [
            "Negative particle surface concentration [mol.m-3]",
            "Electrolyte concentration [mol.m-3]",
            "Positive particle surface concentration [mol.m-3]",
        ]
    )
    plot.dynamic_plot()

# Solve the parameterised battery model with a parameter loop

if False:
    for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
        solution = solver.solve(
            model,
            t_eval,
            inputs={"Current function [A]": value}
        )
        plt.plot(solution.t, solution["Terminal voltage [V]"].data, label=f"{value} A")
    plt.show()

# Optimisation

if False:
    def sum_of_squares(parameters):
        print("solving for Current = ", parameters[0])
        simulation = solver.solve(
            model,
            t_eval,
            inputs={"Current function [A]": parameters[0]}
        )["Terminal voltage [V]"](t_eval)
        return np.sum((simulation - data) ** 2)


    data = solver.solve(
        model,
        t_eval,
        inputs={"Current function [A]": 0.2222}
    )["Terminal voltage [V]"](t_eval)

    bounds = (0.01, 0.6)
    x0 = np.random.uniform(low=bounds[0], high=bounds[1])
    res = scipy.optimize.minimize(sum_of_squares, x0, bounds=[bounds])
    print(res.x[0])

# Provide parameter list to solver

if True:
    model = pybamm.lithium_ion.DFN()
    model.convert_to_format = 'jax'
    model.events = []  # remove events (not supported in jax)
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update({"Current function [A]": "[input]"})
    param.process_geometry(geometry)
    param.process_model(model)

    var = pybamm.standard_spatial_vars
    n = 5
    k = 3
    values = np.linspace(0.1, 0.5, 8)
    var_pts = {var.x_n: n, var.x_s: n, var.x_p: n, var.r_n: k, var.r_p: k}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    t_eval = np.linspace(0, 3600, 100)
    solver = pybamm.JaxSolver(atol=1e-6, rtol=1e-6, method="BDF")
    # solution = solver.solve(
    #     model,
    #     t_eval,
    #     inputs={"Current function [A]": 0.15652}
    #  )

    if False:
        for value in values:
            solution = solver.solve(
                model,
                t_eval,
                inputs={"Current function [A]": value}
            )
            plt.plot(solution.t, solution["Terminal voltage [V]"].data, label=f"{value} A")
        plt.show()
    if True:
        inputs = []
        for value in values:
            inputs.append({"Current function [A]": value})
        solution = solver.solve(
            model,
            t_eval,
            inputs=inputs
        )
        if not isinstance(solution, list):
            solution = [solution]
        for sol in solution:
            plt.plot(sol.t, sol["Terminal voltage [V]"].data)
        plt.show()
