import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# set order of Spectral Volume method
order = 3

# load model
# don't use new_copy
models = [pybamm.lithium_ion.DFN(name="Finite Volume"),
          pybamm.lithium_ion.DFN(name="Spectral Volume")]

# create geometry
geometries = [m.default_geometry for m in models]

# load parameter values and process model and geometry
params = [m.default_parameter_values for m in models]
for m, p, g in zip(models, params, geometries):
    p.process_model(m)
    p.process_geometry(g)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 1, var.x_s: 1, var.x_p: 1, var.r_n: 1, var.r_p: 1}
# the Finite Volume method also works on spectral meshes
meshes = [pybamm.Mesh(
    geometry,
    {
        "negative particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "positive particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "negative electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "separator": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "positive electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order}
        ),
        "current collector": pybamm.SubMesh0D,
    },
    var_pts) for geometry in geometries]

# discretise model
disc_fv = pybamm.Discretisation(meshes[0], models[0].default_spatial_methods)
disc_sv = pybamm.Discretisation(
    meshes[1],
    {
        "negative particle": pybamm.SpectralVolume(order=order),
        "positive particle": pybamm.SpectralVolume(order=order),
        "negative electrode": pybamm.SpectralVolume(order=order),
        "separator": pybamm.SpectralVolume(order=order),
        "positive electrode": pybamm.SpectralVolume(order=order),
        "current collector": pybamm.ZeroDimensionalSpatialMethod()
    }
)

disc_fv.process_model(models[0])
disc_sv.process_model(models[1])

# solve model
t_eval = np.linspace(0, 3600, 100)

casadi_fv = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8).solve(models[0], t_eval)
casadi_sv = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8).solve(models[1], t_eval)
solutions = [casadi_fv, casadi_sv]

# plot
plot = pybamm.QuickPlot(solutions)
plot.dynamic_plot()
