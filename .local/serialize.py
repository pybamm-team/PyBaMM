import json

import pybamm

model = pybamm.lithium_ion.SPM({"thermal": "lumped"})

vars_to_keep = [
    "Current [A]",
    "Current variable [A]",
    "Voltage [V]",
    "Power [W]",
    "Time [s]",
    "Battery voltage [V]",
    "Anode potential [V]",
    "Cathode potential [V]",
    "Internal resistance [mΩ]",
    "Open-circuit voltage [V]",
    "State of charge [%]",
    "Negative electrode stoichiometry",
    "Positive electrode stoichiometry",
    "Volume-averaged cell temperature [K]",
    "Total current density [A.m-2]",
    "C-rate",
]

for var in [*model.rhs.keys(), *model.algebraic.keys()]:
    if isinstance(var, pybamm.Variable):
        vars_to_keep.append(var.name)
    elif isinstance(var, pybamm.ConcatenationVariable):
        for subvar in var.children:
            vars_to_keep.append(subvar.name)

model.variables = {k: v for k, v in model.variables.items() if k in vars_to_keep}

# Serialize the model to a JSON-serializable dictionary
serializer = pybamm.Serialise()
model_dict = serializer.serialise_custom_model(model)

# Save to JSON file
filename = "model.json"
with open(filename, "w") as f:
    json.dump(model_dict, f, indent=2, default=pybamm.Serialise._json_encoder)

print(f"Model serialized to {filename}")

# Deserialize and solve the model
print("\nDeserializing model from JSON...")
loaded_model = pybamm.Serialise.load_custom_model(filename)
print(f"Model loaded: {loaded_model.name}")

# Create parameter values (use model's default if available)
param = pybamm.ParameterValues("Chen2020")

# Process model and geometry with parameters
geometry = loaded_model.default_geometry
param.process_model(loaded_model)
param.process_geometry(geometry)

# Create mesh
var_pts = {
    "x_n": 20,
    "x_s": 20,
    "x_p": 20,
    "r_n": 30,
    "r_p": 30,
}
mesh = pybamm.Mesh(geometry, loaded_model.default_submesh_types, var_pts)

# Discretize model
disc = pybamm.Discretisation(mesh, loaded_model.default_spatial_methods)
disc.process_model(loaded_model)

# Create solver and solve
solver = pybamm.CasadiSolver()
t_eval = [0, 8000]  # 1 hour simulation
print("\nSolving model...")
solution = solver.solve(loaded_model, t_eval=t_eval)
print("Solution computed successfully!")
print(f"Termination reason: {solution.termination}")
print(f"Final time: {solution.t[-1]:.2f} s")
print(f"Final voltage: {solution['Voltage [V]'](solution.t[-1]):.3f} V")

solution.plot(["Voltage [V]", "Current [A]"])
