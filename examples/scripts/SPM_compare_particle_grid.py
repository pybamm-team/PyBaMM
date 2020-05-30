#
# Compare different discretisations in the particle
#
import argparse
import numpy as np
import pybamm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action="store_true", help="Set logging level to 'DEBUG'."
)
args = parser.parse_args()
if args.debug:
    pybamm.set_logging_level("DEBUG")
else:
    pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPM(name="Uniform mesh"),
    pybamm.lithium_ion.SPM(name="Chebyshev mesh"),
    pybamm.lithium_ion.SPM(name="Exponential mesh"),
]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
for model in models:
    param.process_model(model)

# set mesh
submesh_types = models[0].default_submesh_types
particle_meshes = [
    pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
    pybamm.MeshGenerator(pybamm.Chebyshev1DSubMesh),
    pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}),
]
meshes = [None] * len(models)
# discretise models
for i, model in enumerate(models):
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    submesh_types["negative particle"] = particle_meshes[i]
    submesh_types["positive particle"] = particle_meshes[i]
    meshes[i] = pybamm.Mesh(geometry, submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(meshes[i], model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 3600, 100)
for i, model in enumerate(models):
    solutions[i] = model.default_solver.solve(model, t_eval)

# process particle concentration variables
processed_variables = [None] * len(models)
for i, solution in enumerate(solutions):
    c_n = solution["X-averaged negative particle concentration [mol.m-3]"]
    c_p = solution["X-averaged positive particle concentration [mol.m-3]"]
    processed_variables[i] = {"c_n": c_n, "c_p": c_p}


# plot
def plot(t):
    plt.subplots(figsize=(15, 8))
    plt.subplot(121)
    plt.xlabel(r"$r_n$")
    plt.ylabel("Negative particle concentration [mol.m-3]")
    for i, variables in enumerate(processed_variables):
        r_n = solutions[i]["r_n [m]"].entries[:, 0, 0]
        plt.plot(r_n, variables["c_n"](r=r_n, t=t), "-o", label=models[i].name)
    plt.subplot(122)
    plt.xlabel(r"$r_p$")
    plt.ylabel("Positive particle concentration [mol.m-3]")
    for i, variables in enumerate(processed_variables):
        r_p = solutions[i]["r_p [m]"].entries[:, 0, 0]
        plt.plot(r_p, variables["c_p"](r=r_p, t=t), "-o", label=models[i].name)
    plt.legend()
    plt.show()


plot(600)
