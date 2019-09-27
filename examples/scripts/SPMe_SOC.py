import pybamm
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
pybamm.set_logging_level("INFO")

factor = 6.38

# Dimensions
h = 0.137
w = 0.207/factor
A = h * w
l_n = 1e-4
l_p = 1e-4
l_s = 2.5e-5
l1d = (l_n + l_p + l_s)
vol = h * w * l1d
vol_cm3 = vol * 1e6

tot_cap = 0.0
tot_time = 0.0
fig, axes = plt.subplots(1, 2, sharey=True)
I_mag = 1.01/factor
for enum, I_app in enumerate([-1.0, 1.0]):
    I_app *= I_mag
    # load model
    model = pybamm.lithium_ion.SPMe()
    # create geometry
    geometry = model.default_geometry
    # load parameter values and process model and geometry
    param = model.default_parameter_values

    param.update(
        {
            "Electrode height [m]": h,
            "Electrode width [m]": w,
            "Negative electrode thickness [m]": l_n,
            "Positive electrode thickness [m]": l_p,
            "Separator thickness [m]": l_s,
            "Lower voltage cut-off [V]": 3.105,
            "Upper voltage cut-off [V]": 4.7,
            "Maximum concentration in negative electrode [mol.m-3]": 25000,
            "Maximum concentration in positive electrode [mol.m-3]": 50000,
            "Initial concentration in negative electrode [mol.m-3]": 12500,
            "Initial concentration in positive electrode [mol.m-3]": 25000,
            "Negative electrode surface area density [m-1]": 180000.0,
            "Positive electrode surface area density [m-1]": 150000.0,
            "Typical current [A]": I_app,
        }
    )

    param.process_model(model)
    param.process_geometry(geometry)
    s_var = pybamm.standard_spatial_vars
    var_pts = {s_var.x_n: 5, s_var.x_s: 5, s_var.x_p: 5,
               s_var.r_n: 5, s_var.r_p: 10}
    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    # solve model
    t_eval = np.linspace(0, 0.2, 100)
    sol = model.default_solver.solve(model, t_eval)

    var = "Positive electrode average extent of lithiation"
    xpext = pybamm.ProcessedVariable(model.variables[var],
                                     sol.t, sol.y, mesh=mesh)
    var = "Negative electrode average extent of lithiation"
    xnext = pybamm.ProcessedVariable(model.variables[var],
                                     sol.t, sol.y, mesh=mesh)
    time = pybamm.ProcessedVariable(model.variables["Time [h]"],
                                    sol.t, sol.y, mesh=mesh)

    # Coulomb counting
    time_hours = time(sol.t)
    dc_time = np.around(time_hours[-1], 3)
    # Capacity mAh
    cap = np.absolute(I_app * 1000 * dc_time)

    axes[enum].plot(np.absolute(I_app)*1000*time_hours,
                    xnext(sol.t), 'r-', label='Negative')
    axes[enum].plot(np.absolute(I_app)*1000*time_hours,
                    xpext(sol.t), 'b-', label='Positive')
    axes[enum].set_xlabel('Capacity [mAh]')
    plt.legend()
    if I_app < 0.0:
        axes[enum].set_ylabel('Extent of Lithiation')
        axes[enum].title.set_text('Charge')
    else:
        axes[enum].title.set_text('Discharge')
    print('Applied Current', I_app, 'A', 'Time',
          dc_time, 'hrs', 'Capacity', cap, 'mAh')
    tot_cap += cap
    tot_time += dc_time

print('Total Charge/Discharge Time', tot_time, 'hrs')
print('Total Capacity', np.around(tot_cap, 3), 'mAh')
print('Total Capacity', np.around(tot_cap, 3)/vol_cm3, 'mAh.cm-3')
