import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models

thermal = True
C_rate = 1
t_eval = np.linspace(0, 0.17, 100)

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
}

# models = {
#     "SPM": models.solve_spm(C_rate, t_eval, var_pts),
#     "SPMeCC": models.solve_spmecc(C_rate, t_eval, var_pts),
#     "DFN": models.solve_spmecc(C_rate, t_eval, var_pts),
#     "DFNCC": models.solve_spmecc(C_rate, t_eval, var_pts),
#     "2+1D SPM": models.solve_spmecc(C_rate, t_eval, var_pts),
#     "2+1D SPMe": models.solve_spmecc(C_rate, t_eval, var_pts),
#     "2+1D DFN": models.solve_spmecc(C_rate, t_eval, var_pts),
# }


models = {
    "SPM": models.SPM(thermal),
    "DFN": models.DFN(thermal),
    "2+1D DFN": models.DFN_2p1D(thermal),
}

linestyles = {"SPM": ":", "SPMeCC": "-.", "DFN": "--", "2+1D DFN": "-"}

colors = {"SPM": "b", "DFN": "r", "2+1D DFN": "g"}


for name, model, in models.items():

    model.solve(var_pts, C_rate, t_eval)

    variables = ["Discharge capacity [A.h]", "Terminal voltage [V]"]
    pv = model.processed_variables(variables)

    plt.plot(
        pv["Discharge capacity [A.h]"](model.t),
        pv["Terminal voltage [V]"](model.t),
        label=name,
        linestyle=linestyles[name],
        color=colors[name],
    )

plt.xlabel("Discharge capacity [A.h]")
plt.ylabel("Terminal voltage [V]")
plt.legend()
plt.show()

