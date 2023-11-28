#
# Simulate insertion of a reference electrode in the middle of the cell
#
import pybamm

# load model
model = pybamm.lithium_ion.SPM()

# load parameters and evaluate the mid-point of the cell
parameter_values = pybamm.ParameterValues("Chen2020")
L_n = model.param.n.L
L_s = model.param.s.L
L_mid = parameter_values.evaluate(L_n + L_s / 2)

# extract the potential in the negative and positive electrode at the electrode/current
# collector interfaces
phi_n = pybamm.boundary_value(
    model.variables["Negative electrode potential [V]"], "left"
)
phi_p = pybamm.boundary_value(
    model.variables["Positive electrode potential [V]"], "right"
)

# evaluate the electrolyte potential at the mid-point of the cell
phi_e_mid = pybamm.EvaluateAt(model.variables["Electrolyte potential [V]"], L_mid)

# add the new variables to the model
model.variables.update(
    {
        "Negative electrode 3E potential [V]": phi_n - phi_e_mid,
        "Positive electrode 3E potential [V]": phi_p - phi_e_mid,
    }
)

# solve
sim = pybamm.Simulation(model)
sim.solve([0, 3600])

# plot a comparison of the 3E potential and the potential difference between the solid
# and electrolyte phases at the electrode/separator interfaces
sim.plot(
    [
        [
            "Negative electrode surface potential difference at separator interface [V]",
            "Negative electrode 3E potential [V]",
        ],
        [
            "Positive electrode surface potential difference at separator interface [V]",
            "Positive electrode 3E potential [V]",
        ],
        "Voltage [V]",
    ]
)
