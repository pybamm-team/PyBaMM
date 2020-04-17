#
# Compare pybamm lithium-ion battery model with comsol model
#
import pybamm
import numpy as np
import os
import pickle
import scipy.interpolate as interp

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model (x-full refers to solving the full PDE for T)
model = pybamm.lithium_ion.DFN({"thermal": "x-full"}, name="PyBaMM")

# load parameter values and process model
param = model.default_parameter_values
C_rate = 1
param.update({"C-rate": C_rate})
param.process_model(model)

# create geometry
geometry = model.default_geometry
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: int(param.evaluate(pybamm.geometric_parameters.L_n / 1e-6)),
    var.x_s: int(param.evaluate(pybamm.geometric_parameters.L_s / 1e-6)),
    var.x_p: int(param.evaluate(pybamm.geometric_parameters.L_p / 1e-6)),
    var.r_n: int(param.evaluate(pybamm.geometric_parameters.R_n / 1e-7)),
    var.r_p: int(param.evaluate(pybamm.geometric_parameters.R_p / 1e-7)),
}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_eval = np.linspace(0, 3600 / tau / C_rate, 60)
solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, mode="fast")
solution = solver.solve(model, t_eval)

# variables for plotting
output_variables = [
    "Negative electrode potential [V]",
    "Positive electrode potential [V]",
    "Negative electrode current density [A.m-2]",
    "Positive electrode current density [A.m-2]",
    "Electrolyte concentration [mol.m-3]",
    "Electrolyte potential [V]",
    "Terminal voltage [V]",
    "Volume-averaged cell temperature [K]",
]

"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"


def make_comsol_model(comsol_variables, name):
    whole_cell = ["negative electrode", "separator", "positive electrode"]
    comsol_t = comsol_variables["time"]
    L_x = param.evaluate(pybamm.standard_parameters_lithium_ion.L_x)
    interp_kind = "cubic"

    def get_interp_fun(variable_name, domain):
        """
        Create a :class:`pybamm.Function` object using the variable, to allow plotting
        with :class:`'pybamm.QuickPlot'` (interpolate in space to match edges, and
        then create function to interpolate in time)
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
        variable = interp.interp1d(comsol_x, variable, axis=0, kind=interp_kind)(
            pybamm_x
        )

        def myinterp(t):
            return interp.interp1d(comsol_t, variable, kind=interp_kind)(t)[
                :, np.newaxis
            ]

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
    comsol_voltage = interp.interp1d(
        comsol_t, comsol_variables["voltage"], kind=interp_kind
    )
    try:
        comsol_temperature_av = interp.interp1d(
            comsol_t, comsol_variables["average temperature"], kind=interp_kind
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
        "Terminal voltage [V]": pybamm.Function(
            comsol_voltage, pybamm.t * tau, name="voltage_comsol"
        ),
        "Volume-averaged cell temperature [K]": pybamm.Function(
            comsol_temperature_av, pybamm.t * tau, name="temperature_comsol"
        ),
    }

    return comsol_model


comsol_variables = pickle.load(
    open("input/comsol_results/comsol_thermal_1C.pickle", "rb")
)
comsol_model = make_comsol_model(comsol_variables, "COMSOL")

"-----------------------------------------------------------------------------"
"Make plots"

plot = pybamm.QuickPlot(
    [model, comsol_model],
    mesh,
    [solution, solution],
    output_variables=output_variables,
    linestyles=["-", ":"],
)
plot.dynamic_plot()
