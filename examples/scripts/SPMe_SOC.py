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
vol = h * w * l1d # m^3
vol_cm3 = vol * 1e6

tot_cap = 0.0
tot_time = 0.0

I_mag = 862e-3
for I_app in [-1.0, 1.0]:
    I_app*=I_mag
    # load model
    model = pybamm.lithium_ion.SPMe()
    # create geometry
    geometry = model.default_geometry
    # load parameter values and process model and geometry
    param = model.default_parameter_values
    #I_app = 1.0 # [A]
    param.update(
        {
#            "Electrode height [m]": h,
#            "Electrode width [m]": w,
            "Negative electrode thickness [m]": l_n,
            "Positive electrode thickness [m]": l_p,
            "Separator thickness [m]": l_s,
            "Lower voltage cut-off [V]": 3.105,
            "Upper voltage cut-off [V]": 4.7,
            "Maximum concentration in negative electrode [mol.m-3]": 24983.261993843702,
            "Maximum concentration in positive electrode [mol.m-3]": 51217.9257309275,
            "Negative electrode surface area density [m-1]": 180000.0,
            "Positive electrode surface area density [m-1]": 150000.0,
            "Typical current [A]": I_app,
        }
    )

#    max_neg = param["Maximum concentration in negative electrode [mol.m-3]"]
#    max_pos = param["Maximum concentration in positive electrode [mol.m-3]"]
#    init_neg = param["Initial concentration in negative electrode [mol.m-3]"]
#    init_pos = param["Initial concentration in positive electrode [mol.m-3]"]
#    a_neg = param["Negative electrode surface area density [m-1]"]
#    a_pos = param["Positive electrode surface area density [m-1]"]
#    l_neg = param["Negative electrode thickness [m]"]
#    l_pos = param["Positive electrode thickness [m]"]
#    l_sep = param["Separator thickness [m]"]
#    por_neg = param["Negative electrode porosity"]
#    por_pos = param["Positive electrode porosity"]
#    por_sep = param["Separator porosity"]
    
    param.process_model(model)
    param.process_geometry(geometry)
    s_var = pybamm.standard_spatial_vars
    var_pts = {s_var.x_n: 5, s_var.x_s: 5, s_var.x_p: 5, s_var.r_n: 5, s_var.r_p: 10}
    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    # solve model
    t_eval = np.linspace(0, 0.2, 100)
    sol = model.default_solver.solve(model, t_eval)
    
    # plot
    vars = [
        "Electrolyte concentration",
        "Positive electrode volume-averaged concentration [mol.m-3]",
        "Negative electrode volume-averaged concentration [mol.m-3]",
        "Positive electrode average extent of lithiation",
        "Negative electrode average extent of lithiation",
        "X-averaged positive electrolyte concentration [mol.m-3]",
        "X-averaged negative electrolyte concentration [mol.m-3]",
        "X-averaged separator electrolyte concentration [mol.m-3]"
    ]
#    plot = pybamm.QuickPlot(model, mesh, sol, output_variables=vars)
#    plot.dynamic_plot()
#    keys = list(model.variables.keys())
#    keys.sort()
    
#    avol_pos = pybamm.ProcessedVariable(model.variables[vars[0]], sol.t, sol.y, mesh=mesh)
#    avol_neg = pybamm.ProcessedVariable(model.variables[vars[1]], sol.t, sol.y, mesh=mesh)
#    xppc = pybamm.ProcessedVariable(model.variables[vars[2]], sol.t, sol.y, mesh=mesh)
#    xnpc = pybamm.ProcessedVariable(model.variables[vars[3]], sol.t, sol.y, mesh=mesh)
    xpext = pybamm.ProcessedVariable(model.variables[vars[3]], sol.t, sol.y, mesh=mesh)
    xnext = pybamm.ProcessedVariable(model.variables[vars[4]], sol.t, sol.y, mesh=mesh)
#    xppce = pybamm.ProcessedVariable(model.variables[vars[6]], sol.t, sol.y, mesh=mesh)
#    xnpce = pybamm.ProcessedVariable(model.variables[vars[7]], sol.t, sol.y, mesh=mesh)
#    xsepe = pybamm.ProcessedVariable(model.variables[vars[8]], sol.t, sol.y, mesh=mesh)
    time = pybamm.ProcessedVariable(model.variables["Time [h]"], sol.t, sol.y, mesh=mesh)
#    rp = np.linspace(0, 1.0, 11)
#        
#    plt.figure()
#    rplt = 0.0
#    plt.plot(avol_neg(sol.t, r=rplt) * max_neg, "r--", label="Max Neg")
#    plt.plot(avol_neg(sol.t, r=rplt) * init_neg, "r*-", label="Init Neg")
#    plt.plot(avol_pos(sol.t, r=rplt) * max_pos, "b--", label="Max Pos")
#    plt.plot(avol_pos(sol.t, r=rplt) * init_pos, "b*-", label="Init Pos")
#    plt.plot(xnpc(sol.t, r=rplt), "r", label="Neg Li")
#    plt.plot(xppc(sol.t, r=rplt), "b", label="Pos Li")
#    
#    pos_electrolyte_li = xppce(sol.t)*l_pos*por_pos
#    neg_electrolyte_li = xnpce(sol.t)*l_neg*por_neg
#    sep_electrolyte_li = xsepe(sol.t)*l_sep*por_sep
#    tot_electrolyte_li = pos_electrolyte_li+neg_electrolyte_li+sep_electrolyte_li
#    plt.plot(tot_electrolyte_li, "g", label="Elec Li")
#    tot_li = xnpc(sol.t, r=rplt) + xppc(sol.t, r=rplt) + tot_electrolyte_li
#    plt.plot(tot_li, "k-", label="Total Li")
#    plt.legend()
    
    # Coulomb counting
    time_hours = time(sol.t)
    dc_time = np.around(time_hours[-1], 3)
    cap = np.absolute(I_app * 1000 * dc_time) # mAh
    plt.figure()
    plt.plot(np.absolute(I_app)*1000*time_hours, xnext(sol.t), 'r-', label='Negative')
    plt.plot(np.absolute(I_app)*1000*time_hours, xpext(sol.t), 'b-', label='Positive')
    plt.xlabel('Capacity [mAh]')
    plt.ylabel('Extent of Lithiation')
    plt.legend()
    if I_app < 0.0:
        plt.title('Charge')
    else:
        plt.title('Discharge')
    print('Applied Current', I_app, 'A', 'Time', dc_time, 'hrs', 'Capacity', cap, 'mAh')
    tot_cap += cap
    tot_time += dc_time

print('Total Charge/Discharge Time', tot_time, 'hrs')
print('Total Capacity', np.around(tot_cap, 3), 'mAh')
print('Total Capacity', np.around(tot_cap, 3)/vol_cm3, 'mAh.cm-3')


