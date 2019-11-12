#
# Compare thermal and isothermal lithium-ion battery models, with comsol models
#
import pybamm
import numpy as np
import os
import pickle
import scipy.interpolate as interp

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

"-----------------------------------------------------------------------------"
"Create and solve pybamm models"

# load models
options = {"thermal": "isothermal"}
models = [
    pybamm.lithium_ion.DFN({"thermal": "isothermal"}, name="PyBaMM: isothermal"),
    pybamm.lithium_ion.DFN({"thermal": "x-full"}, name="PyBaMM: thermal"),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
C_rate = 1
param.update({"C-rate": C_rate})
for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 101, var.x_s: 31, var.x_p: 101, var.r_n: 31, var.r_p: 31}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model for one hour
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
solutions = [None] * len(models)
t_eval = np.linspace(0, 3600 / tau / C_rate, 60)
for i, model in enumerate(models):
    solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, mode="fast")
    solutions[i] = solver.solve(model, t_eval)

output_variables = [
    "Negative electrode potential [V]",
    # "Positive electrode potential [V]",
    "Negative electrode current density [A.m-2]",
    # "Positive electrode current density [A.m-2]",
    "Electrolyte concentration [mol.m-3]",
    "Electrolyte potential [V]",
    # "Terminal voltage [V]",
    # "Volume-averaged cell temperature [K]",
]

"-----------------------------------------------------------------------------"
"Make Comsol 'models' for comparison"


def make_comsol_model(comsol_variables, name):
    whole_cell = ["negative electrode", "separator", "positive electrode"]
    comsol_t = comsol_variables["time"]
    L_x = param.evaluate(pybamm.standard_parameters_lithium_ion.L_x)

    def get_interp_fun(variable_name, domain):
        """
        Create a :class:`pybamm.Function` object using the variable, to allow plotting with
        :class:`'pybamm.QuickPlot'` (interpolate in space to match edges, and then create
        function to interpolate in time)
        """
        variable = comsol_variables[variable_name]
        if domain == ["negative electrode"]:
            comsol_x = comsol_variables["x_n"]
        elif domain == ["separator"]:
            comsol_x = comsol_variables["x_s"]
        elif domain == ["positive electrode"]:
            comsol_x = comsol_variables["x_p"]
        elif domain == whole_cell:
            comsol_x = comsol_variables["x"]
        # Make sure to use dimensional space
        pybamm_x = mesh.combine_submeshes(*domain)[0].nodes * L_x
        variable = interp.interp1d(comsol_x, variable, axis=0)(pybamm_x)

        def myinterp(t):
            return interp.interp1d(comsol_t, variable)(t)[:, np.newaxis]

        # Make sure to use dimensional time
        fun = pybamm.Function(myinterp, pybamm.t * tau, name=variable_name + "_comsol")
        fun.domain = domain
        return fun

    comsol_c_n_surf = get_interp_fun("c_n_surf", ["negative electrode"])
    comsol_c_e = get_interp_fun("c_e", whole_cell)
    comsol_c_p_surf = get_interp_fun("c_p_surf", ["positive electrode"])
    comsol_phi_n = get_interp_fun("phi_n", ["negative electrode"])
    comsol_phi_e = get_interp_fun("phi_e", whole_cell)
    comsol_phi_p = get_interp_fun("phi_p", ["positive electrode"])
    comsol_i_s_n = get_interp_fun("i_s_n", ["negative electrode"])
    comsol_i_s_p = get_interp_fun("i_s_p", ["positive electrode"])
    comsol_i_e_n = get_interp_fun("i_e_n", ["negative electrode"])
    comsol_i_e_p = get_interp_fun("i_e_p", ["positive electrode"])
    comsol_voltage = interp.interp1d(comsol_t, comsol_variables["voltage"])
    try:
        comsol_temperature_av = interp.interp1d(
            comsol_t, comsol_variables["average temperature"]
        )
    except KeyError:
        # isothermal
        def comsol_temperature_av(t):
            return param.evaluate(pybamm.thermal_parameters.T_ref)

    # Create comsol model with dictionary of Matrix variables
    comsol_model = pybamm.BaseModel(name=name)
    comsol_model.variables = {
        "Negative particle surface concentration [mol.m-3]": comsol_c_n_surf,
        "Electrolyte concentration [mol.m-3]": comsol_c_e,
        "Positive particle surface concentration [mol.m-3]": comsol_c_p_surf,
        "Negative electrode potential [V]": comsol_phi_n,
        "Electrolyte potential [V]": comsol_phi_e,
        "Positive electrode potential [V]": comsol_phi_p,
        "Negative electrode current density [A.m-2]": comsol_i_s_n,
        "Positive electrode current density [A.m-2]": comsol_i_s_p,
        "Negative electrode electrolyte current density [A.m-2]": comsol_i_e_n,
        "Positive electrode electrolyte current density [A.m-2]": comsol_i_e_p,
        "Terminal voltage [V]": pybamm.Function(comsol_voltage, pybamm.t * tau),
        "Volume-averaged cell temperature [K]": pybamm.Function(
            comsol_temperature_av, pybamm.t * tau
        ),
    }

    return comsol_model


# Isothermal model
comsol_variables = pickle.load(
    open("input/comsol_results/comsol_isothermal_1C.pickle", "rb")
)
comsol_model_isothermal = make_comsol_model(comsol_variables, "COMSOL: isothermal")
models.append(comsol_model_isothermal)
solutions.append(solutions[0])

# Thermal model
comsol_variables = pickle.load(
    open("input/comsol_results/comsol_thermal_1C.pickle", "rb")
)
comsol_model_thermal = make_comsol_model(comsol_variables, "COMSOL: thermal")
models.append(comsol_model_thermal)
solutions.append(solutions[1])

"-----------------------------------------------------------------------------"
"Make plots"

plot = pybamm.QuickPlot(models, mesh, solutions, output_variables=output_variables)
plot.dynamic_plot()
